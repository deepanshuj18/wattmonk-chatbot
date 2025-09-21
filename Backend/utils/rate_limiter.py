from fastapi import Request
import time
from typing import Dict, Tuple, List, Optional
import asyncio
from utils.error_handlers import RateLimitExceeded
from utils.logging_utils import app_logger

class RateLimiter:
    """
    Simple in-memory rate limiter for API endpoints.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[str, List[float]] = {}
        self.cleanup_task = None
    
    async def start_cleanup_task(self):
        """Start the periodic cleanup task."""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_old_requests())
    
    async def _cleanup_old_requests(self):
        """Periodically clean up old request records."""
        while True:
            try:
                current_time = time.time()
                # Keep only requests from the last minute
                for ip, timestamps in list(self.request_history.items()):
                    self.request_history[ip] = [
                        ts for ts in timestamps if current_time - ts < 60
                    ]
                    # Remove empty entries
                    if not self.request_history[ip]:
                        del self.request_history[ip]
            except Exception as e:
                app_logger.error(f"Error in rate limiter cleanup: {str(e)}")
            
            # Run cleanup every 10 seconds
            await asyncio.sleep(10)
    
    def is_rate_limited(self, ip: str) -> bool:
        """
        Check if the IP is currently rate limited.
        
        Args:
            ip: Client IP address
            
        Returns:
            True if rate limited, False otherwise
        """
        current_time = time.time()
        
        # Initialize history for new IPs
        if ip not in self.request_history:
            self.request_history[ip] = []
        
        # Filter out requests older than 1 minute
        self.request_history[ip] = [
            ts for ts in self.request_history[ip] if current_time - ts < 60
        ]
        
        # Check if rate limit is exceeded
        if len(self.request_history[ip]) >= self.requests_per_minute:
            return True
        
        # Record this request
        self.request_history[ip].append(current_time)
        return False

# Create a global rate limiter instance
rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware to apply rate limiting to API requests.
    
    Args:
        request: FastAPI request
        call_next: Next middleware or endpoint handler
        
    Returns:
        Response from next handler
    """
    # Start the cleanup task if not already running
    await rate_limiter.start_cleanup_task()
    
    # Get client IP
    client_ip = request.client.host
    
    # Skip rate limiting for certain paths
    if request.url.path in ["/", "/api/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Check rate limit
    if rate_limiter.is_rate_limited(client_ip):
        app_logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise RateLimitExceeded("Rate limit exceeded. Please try again later.")
    
    # Process the request
    return await call_next(request)