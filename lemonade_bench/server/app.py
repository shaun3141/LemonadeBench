# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
FastAPI application for the Lemonade Stand Environment.

This module creates an HTTP server that exposes the LemonadeEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Security features:
- Rate limiting to prevent API abuse

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run server
"""

import random
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import Body, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from openenv_core.env_server.http_server import create_app


def generate_seed() -> int:
    """Generate a random seed between 10000 and 99999 (matching web client behavior)."""
    return random.randint(10000, 99999)


# Rate limiting configuration
RATE_LIMIT_REQUESTS = 300  # requests per window
RATE_LIMIT_WINDOW = 1  # window size in seconds


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if a request is allowed for the given client.
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > window_start
        ]
        
        # Check limit
        current_count = len(self.requests[client_id])
        if current_count >= self.max_requests:
            return False, 0
        
        # Record this request
        self.requests[client_id].append(now)
        return True, self.max_requests - current_count - 1


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limiting on API endpoints."""
    
    # Paths to rate limit (nginx strips /api/ prefix, so check direct paths)
    RATE_LIMITED_PATHS = {"/reset", "/step", "/state"}
    
    async def dispatch(self, request: Request, call_next):
        # Rate limit specific API endpoints
        # Note: nginx strips the /api/ prefix, so requests arrive at /reset, /step, etc.
        if request.url.path in self.RATE_LIMITED_PATHS:
            # Use IP address as client identifier
            client_ip = request.client.host if request.client else "unknown"
            
            # Check rate limit
            allowed, remaining = rate_limiter.is_allowed(client_ip)
            
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."},
                    headers={
                        "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(RATE_LIMIT_WINDOW),
                    }
                )
            
            # Process request and add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            return response
        
        return await call_next(request)

# Support both in-repo and standalone imports
try:
    from models import LemonadeAction, LemonadeObservation
    from server.lemonade_environment import LemonadeEnvironment
except ImportError:
    from ..models import LemonadeAction, LemonadeObservation
    from .lemonade_environment import LemonadeEnvironment

# Create the environment instance (will be recreated with seed on reset)
env = LemonadeEnvironment()

# Create the app with web interface and README integration
app = create_app(env, LemonadeAction, LemonadeObservation, env_name="lemonade_bench")

# Remove the default /reset and /step routes from openenv_core so we can override them
# with seed support. The default routes don't handle the seed parameter.
routes_to_remove = ['/reset', '/step']
app.router.routes = [route for route in app.router.routes if getattr(route, 'path', None) not in routes_to_remove]

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)


# Override the reset endpoint to support seed parameter
# NOTE: We register at /reset (not /api/reset) because nginx strips the /api/ prefix
# when proxying from /api/ to the backend server
@app.post("/reset")
async def reset_with_seed(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Reset endpoint with seed support.
    
    Request body can include:
    - seed: Optional integer seed for deterministic runs
    
    If no seed is provided, one is automatically generated. This ensures all games
    are deterministic and reproducible - the seed is always returned in the response.
    
    Returns observation with the seed that was used.
    """
    global env
    
    # Extract seed from request, or generate one if not provided
    # This ensures all games are deterministic and the seed can be used to reproduce results
    seed: int = request.get("seed") if request.get("seed") is not None else generate_seed()
    
    # Create new environment with the seed
    env = LemonadeEnvironment(seed=seed)
    observation = env.reset()
    
    # Serialize observation
    obs_dict = asdict(observation)
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    
    # Include the seed in the response so the UI can display it
    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
        "seed": env._seed,
    }


# Override the step endpoint to match
# NOTE: We register at /step (not /api/step) because nginx strips the /api/ prefix
@app.post("/step")
async def step_action(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step endpoint - executes action and returns observation.
    
    Uses the global environment instance (which has the seed from reset).
    Supports both tier-based (internal) and quantity-based (frontend) APIs.
    """
    global env
    
    # Support both {"action": {...}} and direct action fields
    action_data = request.get("action", request)
    
    # Remove metadata if present
    metadata = action_data.pop("metadata", {})
    
    # Convert quantity-based fields (buy_X) to tier-based if present
    # This allows the frontend to use simpler quantity fields
    from ..models import quantity_to_tier_count
    for supply in ["lemons", "sugar", "cups", "ice"]:
        buy_key = f"buy_{supply}"
        if buy_key in action_data:
            qty = action_data.pop(buy_key)
            if qty and qty > 0:
                tier, count = quantity_to_tier_count(supply, qty)
                action_data[f"{supply}_tier"] = tier
                action_data[f"{supply}_count"] = count
    
    action = LemonadeAction(**action_data)
    action.metadata = metadata
    
    # Execute step
    observation = env.step(action)
    
    # Serialize observation
    obs_dict = asdict(observation)
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    
    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }


def main():
    """
    Entry point for direct execution via uv run or python -m.
    
    This function enables running the server without Docker:
        uv run server
        python -m lemonade_bench.server.app
        openenv serve lemonade_bench
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
