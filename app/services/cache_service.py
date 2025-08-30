import redis
import json
import pickle
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from app.utils.logger import get_logger
from config.settings import Config

logger = get_logger(__name__)

class CacheService:
    def __init__(self):
        try:
            self.redis_client = redis.Redis.from_url(
                Config.REDIS_URL,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            self.redis_client = None
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage
        """
        try:
            # Try JSON first for simple types
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                return json.dumps(value).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(value)
        except Exception as e:
            logger.warning(f"Serialization failed: {str(e)}")
            return pickle.dumps(value)
    
    def _deserialize(self, value: bytes) -> Any:
        """
        Deserialize value from storage
        """
        try:
            # Try JSON first
            return json.loads(value.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                # Fall back to pickle
                return pickle.loads(value)
            except Exception as e:
                logger.warning(f"Deserialization failed: {str(e)}")
                return None
    
    async def get(self, key: str) -> Any:
        """
        Get value from cache
        """
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            return self._deserialize(value)
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, timeout: int = 300) -> bool:
        """
        Set value in cache with optional timeout
        """
        if not self.redis_client:
            return False
        
        try:
            serialized_value = self._serialize(value)
            result = self.redis_client.setex(key, timeout, serialized_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache
        """
        if not self.redis_client:
            return False
        
        try:
            result = self.redis_client.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        """
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
            
        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {str(e)}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache
        """
        if not self.redis_client or not keys:
            return {}
        
        try:
            values = self.redis_client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Cache get_many failed: {str(e)}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], timeout: int = 300) -> bool:
        """
        Set multiple values in cache
        """
        if not self.redis_client or not mapping:
            return False
        
        try:
            pipe = self.redis_client.pipeline()
            
            for key, value in mapping.items():
                serialized_value = self._serialize(value)
                pipe.setex(key, timeout, serialized_value)
            
            results = pipe.execute()
            return all(results)
            
        except Exception as e:
            logger.error(f"Cache set_many failed: {str(e)}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a numeric value in cache
        """
        if not self.redis_client:
            return None
        
        try:
            return self.redis_client.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Cache increment failed for key {key}: {str(e)}")
            return None
    
    async def expire(self, key: str, timeout: int) -> bool:
        """
        Set expiration time for a key
        """
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.expire(key, timeout))
            
        except Exception as e:
            logger.error(f"Cache expire failed for key {key}: {str(e)}")
            return False
    
    async def get_keys_pattern(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern
        """
        if not self.redis_client:
            return []
        
        try:
            keys = self.redis_client.keys(pattern)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
            
        except Exception as e:
            logger.error(f"Cache keys pattern search failed: {str(e)}")
            return []
    
    async def flush_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern
        """
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.get_keys_pattern(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache flush pattern failed: {str(e)}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        if not self.redis_client:
            return {}
        
        try:
            info = self.redis_client.info()
            
            return {
                'redis_version': info.get('redis_version'),
                'used_memory': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses'),
                'uptime_in_seconds': info.get('uptime_in_seconds')
            }
            
        except Exception as e:
            logger.error(f"Cache stats failed: {str(e)}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check
        """
        try:
            start_time = datetime.now()
            
            # Test basic operations
            test_key = f"health_check_{start_time.timestamp()}"
            test_value = {"test": True, "timestamp": start_time.isoformat()}
            
            # Test set
            set_success = await self.set(test_key, test_value, timeout=60)
            
            # Test get
            retrieved_value = await self.get(test_key)
            get_success = retrieved_value == test_value
            
            # Test delete
            delete_success = await self.delete(test_key)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            health_status = {
                'status': 'healthy' if all([set_success, get_success, delete_success]) else 'unhealthy',
                'response_time_ms': response_time,
                'operations': {
                    'set': set_success,
                    'get': get_success,
                    'delete': delete_success
                },
                'timestamp': datetime.now().isoformat()
            }
            
            if self.redis_client:
                stats = await self.get_stats()
                health_status['stats'] = stats
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cache_analysis_result(self, phone_number: str, result: Dict[str, Any], 
                                  cache_type: str = 'fraud_analysis') -> bool:
        """
        Cache fraud analysis result with intelligent expiration
        """
        try:
            cache_key = f"{cache_type}:{phone_number}"
            
            # Determine cache timeout based on risk level
            risk_level = result.get('risk_level', 'UNKNOWN')
            
            if risk_level == 'CRITICAL':
                timeout = 300  # 5 minutes for critical cases
            elif risk_level == 'HIGH':
                timeout = 900  # 15 minutes for high risk
            elif risk_level == 'MEDIUM':
                timeout = 1800  # 30 minutes for medium risk
            else:
                timeout = 3600  # 1 hour for low/unknown risk
            
            # Add cache metadata
            cached_result = {
                **result,
                'cached_at': datetime.now().isoformat(),
                'cache_expires_at': (datetime.now() + timedelta(seconds=timeout)).isoformat(),
                'cache_type': cache_type
            }
            
            return await self.set(cache_key, cached_result, timeout=timeout)
            
        except Exception as e:
            logger.error(f"Cache analysis result failed: {str(e)}")
            return False
    
    async def get_cached_analysis(self, phone_number: str, 
                                cache_type: str = 'fraud_analysis') -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result with freshness check
        """
        try:
            cache_key = f"{cache_type}:{phone_number}"
            result = await self.get(cache_key)
            
            if result:
                # Check if cache is still fresh
                cached_at = datetime.fromisoformat(result.get('cached_at', ''))
                
                # Consider cache stale after certain conditions
                if 'risk_level' in result:
                    max_age_minutes = {
                        'CRITICAL': 5,
                        'HIGH': 15,
                        'MEDIUM': 30,
                        'LOW': 60
                    }.get(result['risk_level'], 30)
                    
                    age_minutes = (datetime.now() - cached_at).total_seconds() / 60
                    
                    if age_minutes > max_age_minutes:
                        # Cache is stale, delete it
                        await self.delete(cache_key)
                        return None
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Get cached analysis failed: {str(e)}")
            return None
    
    async def cache_model_prediction(self, feature_hash: str, prediction: Any, 
                                   model_name: str) -> bool:
        """
        Cache ML model prediction with feature hash
        """
        try:
            cache_key = f"model_prediction:{model_name}:{feature_hash}"
            
            cached_prediction = {
                'prediction': prediction,
                'model_name': model_name,
                'feature_hash': feature_hash,
                'cached_at': datetime.now().isoformat()
            }
            
            # Cache model predictions for 1 hour
            return await self.set(cache_key, cached_prediction, timeout=3600)
            
        except Exception as e:
            logger.error(f"Cache model prediction failed: {str(e)}")
            return False
    
    async def get_cached_prediction(self, feature_hash: str, 
                                  model_name: str) -> Optional[Any]:
        """
        Get cached model prediction
        """
        try:
            cache_key = f"model_prediction:{model_name}:{feature_hash}"
            result = await self.get(cache_key)
            
            if result:
                return result.get('prediction')
            
            return None
            
        except Exception as e:
            logger.error(f"Get cached prediction failed: {str(e)}")
            return None