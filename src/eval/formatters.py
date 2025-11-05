"""
Output formatters for evaluation results.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import json
import pandas as pd
from ..core.interfaces import AbstractOutputFormatter
from ..utils.io import ensure_dir


class CSVFormatter(AbstractOutputFormatter):
    """CSV output formatter."""
    
    @property
    def format_name(self) -> str:
        return "csv"
    
    def format(
        self,
        results: Any,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Format results as CSV.
        
        Args:
            results: DataFrame or dict to format
            output_path: Optional path to save CSV file
            
        Returns:
            CSV string if output_path is None, otherwise None
        """
        if isinstance(results, pd.DataFrame):
            df = results
        elif isinstance(results, dict):
            df = pd.DataFrame([results])
        else:
            raise TypeError(f"Cannot format {type(results)} as CSV")
        
        if output_path:
            path = Path(output_path)
            ensure_dir(path.parent)
            df.to_csv(path, index=False)
            return None
        else:
            return df.to_csv(index=False)


class JSONFormatter(AbstractOutputFormatter):
    """JSON output formatter."""
    
    @property
    def format_name(self) -> str:
        return "json"
    
    def format(
        self,
        results: Any,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Format results as JSON.
        
        Args:
            results: DataFrame or dict/list to format
            output_path: Optional path to save JSON file
            
        Returns:
            JSON string if output_path is None, otherwise None
        """
        if isinstance(results, pd.DataFrame):
            data = results.to_dict(orient="records")
        elif isinstance(results, (dict, list)):
            data = results
        else:
            raise TypeError(f"Cannot format {type(results)} as JSON")
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if output_path:
            path = Path(output_path)
            ensure_dir(path.parent)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return None
        else:
            return json_str


class JSONLFormatter(AbstractOutputFormatter):
    """JSONL (JSON Lines) output formatter."""
    
    @property
    def format_name(self) -> str:
        return "jsonl"
    
    def format(
        self,
        results: Any,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Format results as JSONL.
        
        Args:
            results: DataFrame or list of dicts to format
            output_path: Optional path to save JSONL file
            
        Returns:
            JSONL string if output_path is None, otherwise None
        """
        if isinstance(results, pd.DataFrame):
            records = results.to_dict(orient="records")
        elif isinstance(results, list):
            records = results
        elif isinstance(results, dict):
            records = [results]
        else:
            raise TypeError(f"Cannot format {type(results)} as JSONL")
        
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        jsonl_str = "\n".join(lines)
        
        if output_path:
            path = Path(output_path)
            ensure_dir(path.parent)
            with open(path, "w", encoding="utf-8") as f:
                f.write(jsonl_str)
            return None
        else:
            return jsonl_str


# Registry of available formatters
FORMATTERS_REGISTRY: dict[str, AbstractOutputFormatter] = {
    "csv": CSVFormatter(),
    "json": JSONFormatter(),
    "jsonl": JSONLFormatter(),
}


def get_formatter(format_name: str) -> AbstractOutputFormatter:
    """Get formatter by name."""
    format_lower = format_name.lower()
    if format_lower not in FORMATTERS_REGISTRY:
        raise ValueError(
            f"Unknown format: {format_name}. Available: {list(FORMATTERS_REGISTRY.keys())}"
        )
    return FORMATTERS_REGISTRY[format_lower]

