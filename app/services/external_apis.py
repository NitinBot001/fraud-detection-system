import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from app.utils.logger import get_logger
from app.services.cache_service import CacheService
from config.settings import Config

logger = get_logger(__name__)

class ExternalAPIService:
    def __init__(self):
        self.cache = CacheService()
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # API configurations
        self.apis = {
            'government_fraud_db': {
                'base_url': 'https://api.government-fraud-db.com/v1',
                'api_key': Config.GOVT_API_KEY,
                'rate_limit': 1000,  # requests per hour
                'cache_ttl': 3600
            },
            'telecom_operator': {
                'base_url': 'https://api.telecom-operator.com/v2',
                'api_key': Config.TELECOM_API_KEY,
                'rate_limit': 5000,
                'cache_ttl': 1800
            },
            'ip_geolocation': {
                'base_url': 'https://api.ipgeolocation.io/ipgeo',
                'api_key': 'your_ip_geo_api_key',
                'rate_limit': 10000,
                'cache_ttl': 7200
            },
            'phone_validation': {
                'base_url': 'https://api.numverify.com/v1',
                'api_key': 'your_numverify_api_key',
                'rate_limit': 1000,
                'cache_ttl': 86400
            }
        }
    
    async def check_government_fraud_database(self, phone_number: str) -> Dict[str, Any]:
        """
        Check phone number against government fraud database
        """
        try:
            cache_key = f"govt_fraud_check:{phone_number}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            api_config = self.apis['government_fraud_db']
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                url = f"{api_config['base_url']}/check"
                headers = {
                    'Authorization': f"Bearer {api_config['api_key']}",
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'phone_number': phone_number,
                    'check_type': 'fraud_history'
                }
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Cache the result
                        await self.cache.set(cache_key, result, timeout=api_config['cache_ttl'])
                        
                        return result
                    else:
                        logger.warning(f"Government fraud DB API returned status {response.status}")
                        return {'found': False, 'error': f"API error: {response.status}"}
        
        except asyncio.TimeoutError:
            logger.error("Government fraud DB API timeout")
            return {'found': False, 'error': 'API timeout'}
        except Exception as e:
            logger.error(f"Government fraud DB API error: {str(e)}")
            return {'found': False, 'error': str(e)}
    
    async def validate_phone_number(self, phone_number: str) -> Dict[str, Any]:
        """
        Validate phone number using external service
        """
        try:
            cache_key = f"phone_validation:{phone_number}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            api_config = self.apis['phone_validation']
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                url = f"{api_config['base_url']}/validate"
                params = {
                    'access_key': api_config['api_key'],
                    'number': phone_number,
                    'format': 1
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Cache the result
                        await self.cache.set(cache_key, result, timeout=api_config['cache_ttl'])
                        
                        return result
                    else:
                        return {'valid': False, 'error': f"API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"Phone validation API error: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    async def get_ip_geolocation(self, ip_address: str) -> Dict[str, Any]:
        """
        Get geolocation data for IP address
        """
        try:
            cache_key = f"ip_geo:{ip_address}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            api_config = self.apis['ip_geolocation']
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                url = api_config['base_url']
                params = {
                    'apiKey': api_config['api_key'],
                    'ip': ip_address,
                    'format': 'json'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Cache the result
                        await self.cache.set(cache_key, result, timeout=api_config['cache_ttl'])
                        
                        return result
                    else:
                        return {'error': f"API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"IP geolocation API error: {str(e)}")
            return {'error': str(e)}
    
    async def check_telecom_operator_data(self, phone_number: str) -> Dict[str, Any]:
        """
        Get data from telecom operator
        """
        try:
            cache_key = f"telecom_data:{phone_number}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            api_config = self.apis['telecom_operator']
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                url = f"{api_config['base_url']}/subscriber/info"
                headers = {
                    'Authorization': f"Bearer {api_config['api_key']}",
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'phone_number': phone_number,
                    'data_types': ['basic_info', 'activity_patterns', 'location_history']
                }
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Cache the result
                        await self.cache.set(cache_key, result, timeout=api_config['cache_ttl'])
                        
                        return result
                    else:
                        return {'error': f"API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"Telecom operator API error: {str(e)}")
            return {'error': str(e)}
    
    async def check_international_fraud_databases(self, phone_number: str) -> List[Dict[str, Any]]:
        """
        Check multiple international fraud databases
        """
        try:
            # List of international fraud databases
            databases = [
                {'name': 'FraudBase_US', 'url': 'https://api.fraudbase-us.com/check'},
                {'name': 'EuroFraud', 'url': 'https://api.eurofraud.eu/verify'},
                {'name': 'APAC_FraudNet', 'url': 'https://api.apacfraud.net/lookup'}
            ]
            
            results = []
            
            # Check all databases concurrently
            tasks = []
            for db in databases:
                task = self._check_single_fraud_database(phone_number, db)
                tasks.append(task)
            
            database_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(database_results):
                if isinstance(result, Exception):
                    results.append({
                        'database': databases[i]['name'],
                        'found': False,
                        'error': str(result)
                    })
                else:
                    results.append({
                        'database': databases[i]['name'],
                        **result
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"International fraud database check error: {str(e)}")
            return []
    
    async def _check_single_fraud_database(self, phone_number: str, database: Dict[str, str]) -> Dict[str, Any]:
        """
        Check a single fraud database
        """
        try:
            cache_key = f"fraud_db_{database['name']}:{phone_number}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                payload = {
                    'phone_number': phone_number,
                    'check_type': 'fraud_history'
                }
                
                async with session.post(database['url'], json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Cache the result
                        await self.cache.set(cache_key, result, timeout=3600)
                        
                        return result
                    else:
                        return {'found': False, 'error': f"API error: {response.status}"}
        
        except Exception as e:
            return {'found': False, 'error': str(e)}
    
    async def enrich_phone_data(self, phone_number: str) -> Dict[str, Any]:
        """
        Enrich phone number data from multiple sources
        """
        try:
            # Collect data from all sources concurrently
            tasks = [
                self.validate_phone_number(phone_number),
                self.check_government_fraud_database(phone_number),
                self.check_telecom_operator_data(phone_number),
                self.check_international_fraud_databases(phone_number)
            ]
            
            validation_result, govt_result, telecom_result, intl_results = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            
            # Compile enriched data
            enriched_data = {
                'phone_number': phone_number,
                'timestamp': datetime.now().isoformat(),
                'validation': validation_result if not isinstance(validation_result, Exception) else {},
                'government_database': govt_result if not isinstance(govt_result, Exception) else {},
                'telecom_data': telecom_result if not isinstance(telecom_result, Exception) else {},
                'international_databases': intl_results if not isinstance(intl_results, Exception) else []
            }
            
            # Calculate enrichment score
            enriched_data['enrichment_score'] = self._calculate_enrichment_score(enriched_data)
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Phone data enrichment failed: {str(e)}")
            return {'phone_number': phone_number, 'error': str(e)}
    
    def _calculate_enrichment_score(self, enriched_data: Dict[str, Any]) -> float:
        """
        Calculate how well the phone number data was enriched
        """
        score = 0.0
        
        # Validation data available
        if enriched_data.get('validation', {}).get('valid'):
            score += 0.25
        
        # Government database data
        if enriched_data.get('government_database', {}).get('found'):
            score += 0.35
        
        # Telecom data
        if 'error' not in enriched_data.get('telecom_data', {}):
            score += 0.25
        
        # International database coverage
        intl_dbs = enriched_data.get('international_databases', [])
        if intl_dbs:
            successful_checks = sum(1 for db in intl_dbs if 'error' not in db)
            score += (successful_checks / len(intl_dbs)) * 0.15
        
        return min(1.0, score)
    
    async def bulk_enrich_phone_data(self, phone_numbers: List[str], 
                                   batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Bulk enrich phone number data
        """
        try:
            enriched_data = []
            
            # Process in batches to respect API rate limits
            for i in range(0, len(phone_numbers), batch_size):
                batch = phone_numbers[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [self.enrich_phone_data(phone) for phone in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch enrichment error: {str(result)}")
                    else:
                        enriched_data.append(result)
                
                # Rate limiting delay between batches
                await asyncio.sleep(1)
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Bulk enrichment failed: {str(e)}")
            return []
    
    async def monitor_api_health(self) -> Dict[str, Any]:
        """
        Monitor health of external APIs
        """
        health_status = {}
        
        for api_name, config in self.apis.items():
            try:
                start_time = datetime.now()
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    # Simple health check endpoint
                    health_url = f"{config['base_url']}/health"
                    
                    async with session.get(health_url) as response:
                        response_time = (datetime.now() - start_time).total_seconds()
                        
                        health_status[api_name] = {
                            'status': 'healthy' if response.status == 200 else 'unhealthy',
                            'response_time_ms': response_time * 1000,
                            'status_code': response.status,
                            'last_checked': datetime.now().isoformat()
                        }
            
            except Exception as e:
                health_status[api_name] = {
                    'status': 'error',
                    'error': str(e),
                    'last_checked': datetime.now().isoformat()
                }
        
        return health_status