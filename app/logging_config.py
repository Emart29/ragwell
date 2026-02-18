import logging
import sys
import uuid
import contextvars
from datetime import datetime

# Context variable for request ID to trace logs across a request
request_id_ctx = contextvars.ContextVar("request_id", default=None)

class RequestIdFilter(logging.Filter):
    """Logging filter to inject request_id into log records."""
    def filter(self, record):
        record.request_id = request_id_ctx.get() or "system"
        return True

def setup_logging():
    """Configure structured logging for Ragwell."""
    log_format = "%(asctime)s | %(levelname)-8s | [%(request_id)s] | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    handler.addFilter(RequestIdFilter())
    
    # Clean up existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(handler)
    
    # Set specific levels for noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    return logger

def get_logger(name):
    """Get a named logger."""
    return logging.getLogger(name)

def generate_request_id():
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:8]
