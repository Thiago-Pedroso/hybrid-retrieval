"""
Sistema de logging avançado para debugging e profiling.

Features:
- Logs coloridos no terminal
- Múltiplos níveis configuráveis (DEBUG, INFO, WARNING, ERROR)
- Logs para arquivo opcional
- Context managers para timing de operações
- Decoradores para profiling de funções
- Configuração via variável de ambiente HYBRID_LOG_LEVEL
"""
import logging
import sys
import time
import functools
from pathlib import Path
from typing import Optional, Callable, Any
from contextlib import contextmanager
from datetime import datetime

# ==================== Cores ANSI ====================
class Colors:
    """Códigos ANSI para colorir output no terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Cores principais
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Cores brilhantes
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Formatter com cores por nível de log."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.BRIGHT_CYAN,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            level_color = self.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{level_color}{record.levelname:8s}{Colors.RESET}"
            record.name = f"{Colors.BRIGHT_BLUE}{record.name}{Colors.RESET}"
            
            # Timestamp em cinza
            timestamp = f"{Colors.DIM}{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}{Colors.RESET}"
            
            # Mensagem
            msg = record.getMessage()
            
            # Localização (arquivo:linha) em dim
            location = f"{Colors.DIM}[{record.filename}:{record.lineno}]{Colors.RESET}"
            
            return f"{timestamp} | {record.levelname} | {record.name:15s} | {location} | {msg}"
        else:
            timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
            return f"{timestamp} | {record.levelname:8s} | {record.name:15s} | [{record.filename}:{record.lineno}] | {record.getMessage()}"


# ==================== Configuração Global ====================
_DEFAULT_LEVEL = logging.INFO
_LOG_DIR: Optional[Path] = None
_FILE_HANDLER: Optional[logging.FileHandler] = None


def set_log_level(level: str | int):
    """Define o nível de log global."""
    global _DEFAULT_LEVEL
    if isinstance(level, str):
        _DEFAULT_LEVEL = getattr(logging, level.upper())
    else:
        _DEFAULT_LEVEL = level
    
    # Atualiza todos os loggers existentes
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        if logger.handlers:
            logger.setLevel(_DEFAULT_LEVEL)


def enable_file_logging(log_dir: str | Path, filename: Optional[str] = None):
    """Habilita logging para arquivo além do console."""
    global _LOG_DIR, _FILE_HANDLER
    
    _LOG_DIR = Path(log_dir)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_file = _LOG_DIR / filename
    
    # Remove handler anterior se existir
    if _FILE_HANDLER:
        for name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(name)
            if _FILE_HANDLER in logger.handlers:
                logger.removeHandler(_FILE_HANDLER)
    
    # Cria novo handler
    _FILE_HANDLER = logging.FileHandler(log_file, encoding='utf-8')
    _FILE_HANDLER.setLevel(logging.DEBUG)  # Arquivo sempre pega tudo
    _FILE_HANDLER.setFormatter(
        logging.Formatter('[%(asctime)s] %(levelname)-8s | %(name)-15s | [%(filename)s:%(lineno)d] | %(message)s')
    )
    
    # Adiciona a todos os loggers existentes
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        if logger.handlers:
            logger.addHandler(_FILE_HANDLER)
    
    print(f"{Colors.GREEN}✓ File logging habilitado: {log_file}{Colors.RESET}")


def get_logger(name: str = "hybrid", level: Optional[str | int] = None) -> logging.Logger:
    """
    Obtém logger configurado com cores e formatação.
    
    Args:
        name: Nome do logger (geralmente nome do módulo)
        level: Nível de log específico (ou usa default global)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Se já tem handlers, retorna (evita duplicação)
    if logger.handlers:
        return logger
    
    # Define nível
    if level is None:
        # Tenta ler de variável de ambiente
        import os
        env_level = os.getenv("HYBRID_LOG_LEVEL", "INFO").upper()
        try:
            level = getattr(logging, env_level)
        except AttributeError:
            level = _DEFAULT_LEVEL
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), _DEFAULT_LEVEL)
    
    logger.setLevel(level)
    
    # Handler para console (com cores)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(use_colors=True))
    logger.addHandler(console_handler)
    
    # Adiciona file handler se habilitado
    if _FILE_HANDLER:
        logger.addHandler(_FILE_HANDLER)
    
    # Evita propagação para root logger
    logger.propagate = False
    
    return logger


# ==================== Context Managers para Timing ====================
@contextmanager
def log_time(logger: logging.Logger, operation: str, level: int = logging.INFO):
    """
    Context manager para medir e logar tempo de operação.
    
    Exemplo:
        with log_time(log, "Loading dataset"):
            dataset = load_data()
    """
    start = time.perf_counter()
    logger.log(level, f"⏱️  {operation} - iniciando...")
    
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        
        # Formata tempo de forma legível
        if elapsed < 1:
            time_str = f"{elapsed*1000:.1f}ms"
        elif elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            time_str = f"{hours}h {minutes}m"
        
        logger.log(level, f"✓ {operation} - concluído em {Colors.GREEN}{time_str}{Colors.RESET}")


class ProgressLogger:
    """Logger para operações com progresso (iterações)."""
    
    def __init__(self, logger: logging.Logger, operation: str, total: int, log_every: int = 100):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.log_every = log_every
        self.start_time = time.perf_counter()
        self.count = 0
    
    def update(self, n: int = 1):
        """Atualiza contador e loga se necessário."""
        self.count += n
        
        if self.count % self.log_every == 0 or self.count == self.total:
            elapsed = time.perf_counter() - self.start_time
            rate = self.count / elapsed if elapsed > 0 else 0
            pct = (self.count / self.total * 100) if self.total > 0 else 0
            # ETA simples
            remaining = max(self.total - self.count, 0)
            eta_sec = (remaining / rate) if rate > 0 else float('inf')
            if eta_sec == float('inf'):
                eta_str = "--"
            elif eta_sec < 60:
                eta_str = f"{eta_sec:.1f}s"
            elif eta_sec < 3600:
                eta_str = f"{int(eta_sec//60)}m {eta_sec%60:.0f}s"
            else:
                eta_str = f"{int(eta_sec//3600)}h {int((eta_sec%3600)//60)}m"
            
            self.logger.info(
                f"{self.operation}: {self.count}/{self.total} ({pct:.1f}%) "
                f"@ {rate:.1f} it/s | ETA {eta_str}"
            )
    
    def __enter__(self):
        self.logger.info(f"🚀 {self.operation} - iniciando ({self.total} items)")
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        rate = self.count / elapsed if elapsed > 0 else 0
        self.logger.info(
            f"✓ {self.operation} - concluído: {self.count} items em {elapsed:.2f}s "
            f"({rate:.1f} it/s)"
        )


# ==================== Decoradores para Profiling ====================
def log_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator para logar entrada/saída de funções.
    
    Exemplo:
        @log_call(log, level=logging.INFO)
        def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        # Obtém logger se não fornecido
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Formata argumentos (trunca se muito grande)
            args_str = ", ".join(repr(a)[:50] for a in args)
            kwargs_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            
            logger.log(level, f"→ Calling {func.__name__}({all_args})")
            
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.log(level, f"← {func.__name__} returned in {elapsed*1000:.1f}ms")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"✗ {func.__name__} failed after {elapsed*1000:.1f}ms: {e}")
                raise
        
        return wrapper
    return decorator


def log_exceptions(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """
    Decorator para logar exceções automaticamente.
    
    Args:
        logger: Logger a usar (ou usa o do módulo)
        reraise: Se True, re-levanta a exceção após logar
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {type(e).__name__}: {e}",
                    exc_info=True
                )
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


# ==================== Helper para Debug de Estruturas ====================
def log_shape(logger: logging.Logger, name: str, obj: Any, level: int = logging.DEBUG):
    """Loga shape/tamanho de objetos (arrays, listas, dicts, DataFrames)."""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, np.ndarray):
        info = f"shape={obj.shape}, dtype={obj.dtype}"
    elif isinstance(obj, pd.DataFrame):
        info = f"shape={obj.shape}, columns={list(obj.columns)}"
    elif isinstance(obj, (list, tuple)):
        info = f"len={len(obj)}, type={type(obj).__name__}"
    elif isinstance(obj, dict):
        info = f"keys={len(obj)}, sample_keys={list(obj.keys())[:5]}"
    else:
        info = f"type={type(obj).__name__}, value={str(obj)[:100]}"
    
    logger.log(level, f"📊 {name}: {info}")


# ==================== Inicialização ====================
# Lê nível de log da variável de ambiente
import os
_env_level = os.getenv("HYBRID_LOG_LEVEL", "INFO").upper()
try:
    _DEFAULT_LEVEL = getattr(logging, _env_level)
except AttributeError:
    _DEFAULT_LEVEL = logging.INFO
