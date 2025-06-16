import logging
import re
from logging import LogRecord
import os

# Define a filter to mask sensitive information in logs
class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive data in log records"""
    
    def __init__(self):
        super().__init__()
        # Patterns to identify API keys, tokens, etc.
        self.patterns = [
            # Match API keys in URLs
            (r'[?&]key=([^&\s]+)', r'?key=***MASKED***'),
            # Google API key format
            (r'AIza[0-9A-Za-z-_]{35}', r'***MASKED***'),
            # JWT tokens
            (r'Bearer\s+[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*', r'Bearer ***MASKED***'),
            # Generic API keys
            (r'[a-zA-Z0-9]{32,}', r'***MASKED***'),
        ]
    
    def filter(self, record: LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern, replacement in self.patterns:
                msg = re.sub(pattern, replacement, msg)
            record.msg = msg
            
        # Also check args if they exist
        if record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    for pattern, replacement in self.patterns:
                        args[i] = re.sub(pattern, replacement, arg)
            record.args = tuple(args)
            
        return True

# Configure root logger
def configure_logging():
    """Configure logging with sensitive data filtering"""
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Set level based on environment or default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add sensitive data filter
    console_handler.addFilter(SensitiveDataFilter())
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Configure third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Add filter to httpx logger as well
    httpx_logger = logging.getLogger("httpx")
    for handler in httpx_logger.handlers:
        handler.addFilter(SensitiveDataFilter())
    
    # Return configured logger
    return root_logger