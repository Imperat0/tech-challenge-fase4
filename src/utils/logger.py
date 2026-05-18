"""Configuração de logging centralizada com suporte a erros detalhados."""

import logging
import os
import traceback
import sys
from pathlib import Path


def setup_logger(name: str = "tech_challenge") -> logging.Logger:
    """
    Configura logger centralizado com suporte a DEBUG detalhado.
    
    Variáveis de ambiente:
    - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (padrão: INFO)
    - DEBUG_ERRORS: 1 para mostrar stack traces completos
    """
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Formato mais detalhado para DEBUG
        if level == logging.DEBUG:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> str:
    """
    Log detalhado de exceção com stack trace.
    
    Args:
        logger: Logger instance
        exc: Exception object
        context: Contexto adicional da exceção
        
    Returns:
        String formatada do erro para exibição ao usuário
    """
    error_msg = f"{exc.__class__.__name__}: {str(exc)}"
    if context:
        error_msg = f"[{context}] {error_msg}"
    
    logger.error(error_msg)
    
    # Log stack trace se DEBUG está ativo
    if logger.level <= logging.DEBUG:
        logger.debug("Stack trace:\n%s", traceback.format_exc())
    
    return error_msg
