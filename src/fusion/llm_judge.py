"""
LLM Judge module for evaluating document effectiveness (0-5 scale).
Evaluates both dense and BM25 top-1 candidates together in a single call.
"""

from __future__ import annotations
import os
import json
import time
import re
import random
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta
from ..utils.logging import get_logger

_log = get_logger("fusion.llm_judge")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
    from openai import RateLimitError
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    RateLimitError = Exception  # Fallback


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return len(text) // 4


def _truncate_text(text: str, max_tokens: int = 2000) -> str:
    """Truncate text to approximately max_tokens."""
    if not text:
        return text
    
    # Rough estimation: try to stay under max_tokens
    estimated_tokens = _estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Truncate to approximately max_tokens
    # Use ~4 chars per token as approximation
    max_chars = max_tokens * 4
    truncated = text[:max_chars]
    
    # Try to end at a sentence boundary if possible
    for sep in ['. ', '.\n', '! ', '?\n', '? ']:
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars * 0.8:  # Only if we keep most of the text
            truncated = truncated[:last_sep + len(sep)]
            break
    
    return truncated


class RateLimiter:
    """
    Rate limiter that monitors OpenAI API rate limit headers and waits when needed.
    
    Tracks:
    - Requests per minute (RPM)
    - Tokens per minute (TPM)
    """
    
    def __init__(self, tier: int = 2, safety_margin: float = 0.1):
        """
        Args:
            tier: OpenAI API tier (1=default, 2=higher limits, etc.)
            safety_margin: Fraction of limit to use before waiting (default: 0.1 = 90% usage)
        """
        self.tier = tier
        self.safety_margin = safety_margin
        
        # Known limits per tier (from OpenAI docs)
        # Tier 2: 10,000 RPM, 1,000,000 TPM
        # Tier 1: 3,500 RPM, 90,000 TPM
        tier_limits = {
            1: {"rpm": 3500, "tpm": 90000},
            2: {"rpm": 10000, "tpm": 1000000},
            3: {"rpm": 10000, "tpm": 2000000},
        }
        
        self.limit_rpm = tier_limits.get(tier, tier_limits[2])["rpm"]
        self.limit_tpm = tier_limits.get(tier, tier_limits[2])["tpm"]
        
        # Current usage tracking
        self.remaining_requests = self.limit_rpm
        self.remaining_tokens = self.limit_tpm
        self.reset_time_requests: Optional[datetime] = None
        self.reset_time_tokens: Optional[datetime] = None
        
        # Request history for tracking
        self.request_times: list[datetime] = []
        self.token_usage: list[tuple[datetime, int]] = []
    
    def update_from_headers(self, headers: Dict[str, str]):
        """Update rate limit state from API response headers."""
        # Parse headers (OpenAI SDK may expose these differently)
        # Headers are usually in response.raw_headers or response.headers
        if isinstance(headers, dict):
            # Try to get rate limit info
            remaining_requests = headers.get("x-ratelimit-remaining-requests")
            remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
            reset_requests = headers.get("x-ratelimit-reset-requests")
            reset_tokens = headers.get("x-ratelimit-reset-tokens")
            
            if remaining_requests:
                try:
                    self.remaining_requests = int(remaining_requests)
                except (ValueError, TypeError):
                    pass
            
            if remaining_tokens:
                try:
                    self.remaining_tokens = int(remaining_tokens)
                except (ValueError, TypeError):
                    pass
            
            if reset_requests:
                try:
                    # Reset time is usually seconds until reset
                    reset_seconds = float(reset_requests)
                    self.reset_time_requests = datetime.now() + timedelta(seconds=reset_seconds)
                except (ValueError, TypeError):
                    pass
            
            if reset_tokens:
                try:
                    reset_seconds = float(reset_tokens)
                    self.reset_time_tokens = datetime.now() + timedelta(seconds=reset_seconds)
                except (ValueError, TypeError):
                    pass
    
    def update_from_response(self, response):
        """Update rate limit state from OpenAI API response object."""
        # OpenAI SDK v1+ exposes headers via response._response (httpx.Response)
        headers = {}
        
        # Try multiple ways to access headers (OpenAI SDK structure)
        if hasattr(response, '_response'):
            # httpx.Response object
            raw_response = response._response
            if hasattr(raw_response, 'headers'):
                headers = dict(raw_response.headers)
        elif hasattr(response, 'headers'):
            headers = dict(response.headers)
        elif hasattr(response, '_raw_response'):
            if hasattr(response._raw_response, 'headers'):
                headers = dict(response._raw_response.headers)
        elif hasattr(response, 'raw_headers'):
            headers = dict(response.raw_headers)
        
        # Also try to get from response metadata if available
        if hasattr(response, '_headers'):
            headers.update(dict(response._headers))
        
        if headers:
            self.update_from_headers(headers)
            _log.debug(f"Rate limit headers: remaining_requests={self.remaining_requests}, remaining_tokens={self.remaining_tokens}")
    
    def should_wait(self, estimated_tokens: int = 100) -> Tuple[bool, float]:
        """
        Check if we should wait before making next request.
        
        Args:
            estimated_tokens: Estimated tokens for next request
        
        Returns:
            Tuple of (should_wait, wait_seconds)
        """
        now = datetime.now()
        
        # Clean old history (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > cutoff]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff]
        
        # Check requests limit
        requests_in_window = len(self.request_times)
        requests_threshold = self.limit_rpm * (1 - self.safety_margin)
        
        if requests_in_window >= requests_threshold:
            # Wait until oldest request expires
            if self.request_times:
                oldest = min(self.request_times)
                wait_until = oldest + timedelta(minutes=1)
                wait_seconds = max(0, (wait_until - now).total_seconds())
                if wait_seconds > 0:
                    return True, wait_seconds
        
        # Check tokens limit
        tokens_in_window = sum(tokens for _, tokens in self.token_usage)
        tokens_threshold = self.limit_tpm * (1 - self.safety_margin)
        
        if tokens_in_window + estimated_tokens >= tokens_threshold:
            # Wait until oldest token usage expires
            if self.token_usage:
                oldest = min(t for t, _ in self.token_usage)
                wait_until = oldest + timedelta(minutes=1)
                wait_seconds = max(0, (wait_until - now).total_seconds())
                if wait_seconds > 0:
                    return True, wait_seconds
        
        # Also check remaining from headers (if available)
        if self.remaining_requests < self.limit_rpm * self.safety_margin:
            if self.reset_time_requests:
                wait_seconds = max(0, (self.reset_time_requests - now).total_seconds())
                if wait_seconds > 0:
                    return True, wait_seconds
        
        if self.remaining_tokens < estimated_tokens:
            if self.reset_time_tokens:
                wait_seconds = max(0, (self.reset_time_tokens - now).total_seconds())
                if wait_seconds > 0:
                    return True, wait_seconds
        
        return False, 0.0
    
    def record_request(self, tokens_used: int = 0):
        """Record that a request was made."""
        now = datetime.now()
        self.request_times.append(now)
        if tokens_used > 0:
            self.token_usage.append((now, tokens_used))
        
        # Decrement from header-based tracking
        if self.remaining_requests > 0:
            self.remaining_requests -= 1
        if self.remaining_tokens > tokens_used:
            self.remaining_tokens -= tokens_used
    
    def wait_if_needed(self, estimated_tokens: int = 100):
        """Wait if rate limit is approaching."""
        should_wait, wait_seconds = self.should_wait(estimated_tokens)
        if should_wait and wait_seconds > 0:
            _log.info(f"⏳ Rate limit approaching. Waiting {wait_seconds:.1f} seconds...")
            time.sleep(wait_seconds)
            _log.info("✅ Resuming requests")


class LLMJudge:
    """
    LLM Judge that evaluates effectiveness of documents on a 0-5 scale.
    Evaluates both dense and BM25 top-1 candidates together in a single call.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 10,
        prompt_template: Optional[str] = None,
        cache_dir: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        max_text_tokens: int = 2000,
        rate_limit_tier: int = 2,
        rate_limit_safety_margin: float = 0.1,
    ):
        """
        Args:
            model: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4o")
            temperature: Temperature for generation (default: 0.0)
            max_tokens: Max tokens for response (default: 10, just enough for "D B")
            prompt_template: Template for prompt (will be provided later)
            cache_dir: Directory for caching evaluations
            timeout: Timeout in seconds
            max_retries: Maximum retries on failure
            api_key: OpenAI API key (optional, can use .env)
            max_text_tokens: Maximum tokens to send for each text (~1-2k)
        """
        if not _HAS_OPENAI:
            raise RuntimeError(
                "OpenAI library not installed. Install with: pip install openai"
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or self._default_prompt_template()
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_text_tokens = max_text_tokens
        
        # API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set in .env or as environment variable."
            )
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            tier=rate_limit_tier,
            safety_margin=rate_limit_safety_margin,
        )
        
        # Cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file = self.cache_dir / "llm_judge_cache.json"
            self._cache = self._load_cache()
        else:
            self._cache = {}
    
    def _default_prompt_template(self) -> str:
        """Default prompt template (will be replaced with actual prompt later)."""
        return """You are an evaluator assessing the retrieval effectiveness of dense
retrieval ( Cosine Distance ) and BM25 retrieval for finding the correct answer .

## Task :
Given a question and two top1 search results ( one from dense retrieval ,
one from BM25 retrieval ) , score each retrieval method from **0 to 5**
based on whether the correct answer is likely to appear in top2,top3 , etc .

### ** Scoring Criteria :**
1. ** Direct hit --> 5 points **
- If the retrieved document directly answers the question , assign **5
points **.

2. ** Good wrong result ( High likelihood correct answer is nearby ) --> 3 -4
points **
- If the top1 result is ** conceptually close ** to the correct answer (e . g . , mentions 
relevant entities , related events , partial answer ) ,it indicates the search method is in the right direction .
- Give **4** if it 's very close , **3** if somewhat close .

3. ** Bad wrong result ( Low likelihood correct answer is nearby ) --> 1 -2
points **
- If the top1 result is ** loosely related but misleading ** ( e . g . ,
shares keywords but changes context ) , correct answers might not be
in top2 , top3 .
- Give **2** if there 's a small chance correct answers are nearby ,
**1** if unlikely .

4. ** Completely off - track --> 0 points **
- If the result is ** totally unrelated ** , it means the retrieval
method is failing .
---
### ** Given Data :**
- ** Question :** {query}

---

### ** Given Data :**
- ** Question :** {query}
- ** dense retrieval Top1 Result :** {dense_text}
- ** BM25 retrieval Top1 Result :** {bm25_text}

---

### ** Output Format :**
Return two integers separated by a space:
- ** First number :** dense retrieval score .
- ** Second number :** BM25 retrieval score .
- Example output : 3 4
** Do not output any other text .**
"""
    
    def _load_cache(self) -> Dict[str, Tuple[int, int]]:
        """Load cache from disk."""
        if not self.cache_dir or not self._cache_file.exists():
            return {}
        
        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            _log.warning(f"Failed to load cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        if not self.cache_dir:
            return
        
        try:
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _log.warning(f"Failed to save cache: {e}")
    
    def _cache_key(self, query_id: str, dense_doc_id: str, bm25_doc_id: str) -> str:
        """Generate cache key."""
        return f"{query_id}::{dense_doc_id}::{bm25_doc_id}"
    
    def _parse_output(self, text: str) -> Optional[Tuple[int, int]]:
        """
        Parse LLM output to extract two integers (Dense score, BM25 score).
        Format: "D B" or "3 4" or "D: 3, B: 4" etc.
        Order: Dense first, BM25 second (as per paper).
        """
        if not text:
            return None
        
        # Try to find two integers in the text
        # Pattern: look for two numbers, possibly separated by space or comma
        # Order: Dense first, BM25 second
        patterns = [
            r'D[:\s]*(\d+)[,\s]+B[:\s]*(\d+)',  # "D: 3, B: 4" or "D 3 B 4" - explicit order
            r'(\d+)\s+(\d+)',  # "3 4" - assume first is Dense, second is BM25
            r'(\d+)[,\s]+(\d+)',  # "3, 4" - assume first is Dense, second is BM25
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.strip())
            if match:
                dense_score = int(match.group(1))
                bm25_score = int(match.group(2))
                
                # Validate range
                if 0 <= dense_score <= 5 and 0 <= bm25_score <= 5:
                    return (dense_score, bm25_score)
        
        return None
    
    def evaluate_pair(
        self,
        query: str,
        dense_text: str,
        bm25_text: str,
        query_id: Optional[str] = None,
        dense_doc_id: Optional[str] = None,
        bm25_doc_id: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Evaluate both documents together and return scores (e_dense, e_bm25).
        
        Args:
            query: Query text
            dense_text: Text from dense retrieval top-1
            bm25_text: Text from BM25 retrieval top-1
            query_id: Query ID for caching
            dense_doc_id: Dense document ID for caching
            bm25_doc_id: BM25 document ID for caching
        
        Returns:
            Tuple of (e_dense, e_bm25) scores in range 0-5
        """
        # Check cache
        if query_id and dense_doc_id and bm25_doc_id:
            cache_key = self._cache_key(query_id, dense_doc_id, bm25_doc_id)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                _log.debug(f"Cache hit for {cache_key}: {cached}")
                return tuple(cached)
        
        # Truncate texts to reduce cost
        dense_text_truncated = _truncate_text(dense_text, self.max_text_tokens)
        bm25_text_truncated = _truncate_text(bm25_text, self.max_text_tokens)
        
        # Format prompt (order: Dense first, BM25 second)
        prompt = self.prompt_template.format(
            query=query,
            dense_text=dense_text_truncated,
            bm25_text=bm25_text_truncated,
        )
        
        # Estimate tokens for this request (rough: ~4 chars per token)
        estimated_tokens = len(prompt) // 4 + self.max_tokens
        
        # Call API with retries and rate limiting
        for attempt in range(self.max_retries):
            try:
                # Check and wait if rate limit is approaching
                self.rate_limiter.wait_if_needed(estimated_tokens)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an evaluator. Output only two integers separated by space (D score first, B score second). Format: 'D B' or '3 4'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                
                # Update rate limiter from response headers
                try:
                    self.rate_limiter.update_from_response(response)
                    _log.debug(
                        f"Rate limit status: requests={self.rate_limiter.remaining_requests}/{self.rate_limiter.limit_rpm}, "
                        f"tokens={self.rate_limiter.remaining_tokens}/{self.rate_limiter.limit_tpm}"
                    )
                except Exception as e:
                    _log.debug(f"Could not update rate limiter from headers: {e}")
                
                # Get tokens used (if available)
                tokens_used = 0
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = getattr(response.usage, 'total_tokens', 0) or 0
                
                # Record request
                self.rate_limiter.record_request(tokens_used)
                
                output_text = response.choices[0].message.content.strip()
                _log.debug(f"LLM output (attempt {attempt + 1}): {output_text}")
                
                # Parse output
                parsed = self._parse_output(output_text)
                if parsed is not None:
                    e_dense, e_bm25 = parsed
                    
                    # Save to cache
                    if query_id and dense_doc_id and bm25_doc_id:
                        cache_key = self._cache_key(query_id, dense_doc_id, bm25_doc_id)
                        self._cache[cache_key] = [e_dense, e_bm25]
                        self._save_cache()
                    
                    return (e_dense, e_bm25)
                else:
                    _log.warning(f"Failed to parse LLM output: {output_text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        raise ValueError(
                            f"Could not parse LLM output after {self.max_retries} attempts. "
                            f"Output: {output_text}"
                        )
            
            except RateLimitError as e:
                # Rate limit error - use exponential backoff
                _log.warning(f"Rate limit error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    backoff = min(2 ** attempt + random.uniform(0, 1), 60)
                    _log.info(f"⏳ Rate limit hit. Waiting {backoff:.1f} seconds before retry...")
                    time.sleep(backoff)
                    continue
                else:
                    raise RuntimeError(
                        f"Rate limit error after {self.max_retries} attempts: {e}"
                    ) from e
            
            except Exception as e:
                _log.warning(f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff for other errors
                    backoff = min(2 ** attempt + random.uniform(0, 1), 10)
                    time.sleep(backoff)
                    continue
                else:
                    raise RuntimeError(
                        f"LLM API call failed after {self.max_retries} attempts: {e}"
                    ) from e
        
        raise RuntimeError("Should not reach here")

