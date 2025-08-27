import time
import hashlib
from functools import wraps
from typing import Dict, Any

_CACHE: Dict[str, Any] = {}
_CACHE_TTL = 300  # 5分間のキャッシュ

def _cache_key(*args: Any, **kwargs: Any) -> str:
    key_string = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_string.encode()).hexdigest()

def memory_cache(ttl_seconds: int = _CACHE_TTL):
    """非同期関数向けのメモリキャッシュデコレータ"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{_cache_key(*args, **kwargs)}"
            current_time = time.time()
            
            if key in _CACHE:
                cached_data, timestamp = _CACHE[key]
                if current_time - timestamp < ttl_seconds:
                    return cached_data
                else:
                    del _CACHE[key]
            
            result = await func(*args, **kwargs)
            _CACHE[key] = (result, current_time)
            return result
        return wrapper
    return decorator