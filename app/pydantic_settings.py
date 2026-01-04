"""
Pydantic Settings for configuration
"""
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

__all__ = ['BaseSettings']