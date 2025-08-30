import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, Mock
import psutil
import threading

class TestPerformanceMetrics:
    @pytest.mark.asyncio
    async def test_api_response_time(self, client, auth_headers):
        """Test API response time under normal load."""
        start_time = time.time()
        
        response = client.post('/api/v1/fraud/analyze',
                             headers=auth_headers,
                             json={'phone_number': '+1234567890'})
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # API should respond within 2 seconds
        assert response_time < 2.0
        assert response.status_code in [200, 400, 500]  # Should not timeout

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, client, auth_headers):
        """Test API performance under concurrent load."""
        async def make_request():
            with client:
                response = client.post('/api/v1/fraud/analyze',
                                     headers=auth_headers,
                                     json={'phone_number': '+1234567890'})
                return response.status_code, time.time()
        
        # Make 10 concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Should handle 10 concurrent requests within 10 seconds
        assert total_time < 10.0
        
        # Count successful responses
        successful = sum(1 for result in results 
                        if not isinstance(result, Exception) and result[0] == 200)
        
        # At least 50% should succeed (depending on rate limiting)
        assert successful >= 5

    @pytest.mark.asyncio
    async def test_database_query_performance(self, db_session):
        """Test database query performance."""
        from app.models.database import PhoneNumber, FraudReport
        
        # Create test data
        phones = []
        for i in range(100):
            phone = PhoneNumber(number=f"+123456789{i:02d}", country_code="1")
            phones.append(phone)
        
        db_session.bulk_save_objects(phones)
        db_session.commit()
        
        # Test query performance
        start_time = time.time()
        
        # Complex query with joins
        results = db_session.query(PhoneNumber).join(
            FraudReport, PhoneNumber.id == FraudReport.phone_number_id, isouter=True
        ).limit(50).all()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Database query should complete within 1 second
        assert query_time < 1.0

    def test_memory_usage(self):
        """Test memory usage during processing."""
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate processing large dataset
        large_dataset = []
        for i in range(10000):
            large_dataset.append({
                'phone_number': f'+123456789{i:04d}',
                'risk_score': i / 10000,
                'features': [j for j in range(50)]  # 50 features per record
            })
        
        # Process the dataset
        processed = []
        for record in large_dataset:
            processed_record = {
                'phone': record['phone_number'],
                'risk': record['risk_score'] * 100,
                'feature_count': len(record['features'])
            }
            processed.append(processed_record)
        
        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del large_dataset, processed
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500
        
        # Memory should be mostly freed after cleanup
        assert final_memory - initial_memory < 100

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance."""
        from app.services.cache_service import CacheService
        
        cache = CacheService()
        
        # Test cache write performance
        start_time = time.time()
        
        for i in range(1000):
            await cache.set(f"test_key_{i}", f"test_value_{i}")
        
        write_time = time.time() - start_time
        
        # Should write 1000 items within 5 seconds
        assert write_time < 5.0
        
        # Test cache read performance
        start_time = time.time()
        
        values = []
        for i in range(1000):
            value = await cache.get(f"test_key_{i}")
            values.append(value)
        
        read_time = time.time() - start_time
        
        # Should read 1000 items within 2 seconds
        assert read_time < 2.0
        
        # Verify all values were retrieved
        assert len(values) == 1000
        assert all(v is not None for v in values)

class TestLoadTesting:
    @pytest.mark.asyncio
    async def test_sustained_load(self, client, auth_headers):
        """Test sustained load over time."""
        duration = 30  # seconds
        requests_per_second = 2
        
        start_time = time.time()
        request_count = 0
        errors = 0
        
        while time.time() - start_time < duration:
            try:
                response = client.post('/api/v1/fraud/analyze',
                                     headers=auth_headers,
                                     json={'phone_number': f'+123456789{request_count:02d}'})
                request_count += 1
                
                if response.status_code >= 400:
                    errors += 1
                
                # Rate limiting
                await asyncio.sleep(1.0 / requests_per_second)
                
            except Exception:
                errors += 1
        
        error_rate = errors / request_count if request_count > 0 else 1
        
        # Error rate should be less than 10%
        assert error_rate < 0.1
        
        # Should process at least 50 requests in 30 seconds
        assert request_count >= 50

    def test_cpu_usage_under_load(self):
        """Test CPU usage under processing load."""
        import multiprocessing
        
        # Get initial CPU usage
        initial_cpu = psutil.cpu_percent(interval=1)
        
        def cpu_intensive_task():
            # Simulate fraud detection processing
            for i in range(100000):
                # Simulate risk calculation
                risk_score = (i * 0.001) % 1.0
                risk_level = "HIGH" if risk_score > 0.8 else "LOW"
                
                # Simulate pattern detection
                patterns = []
                if risk_score > 0.5:
                    patterns.append("suspicious")
                if risk_score > 0.7:
                    patterns.append("high_frequency")
        
        # Run CPU intensive tasks in parallel
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(4)]
            
            # Monitor CPU during execution
            max_cpu = 0
            for _ in range(10):
                cpu_usage = psutil.cpu_percent(interval=0.5)
                max_cpu = max(max_cpu, cpu_usage)
            
            # Wait for tasks to complete
            for future in futures:
                future.result()
        
        # CPU usage should be reasonable (not constantly at 100%)
        assert max_cpu < 95

class TestScalabilityMetrics:
    @pytest.mark.asyncio
    async def test_database_connection_scaling(self, db_session):
        """Test database connection handling under load."""
        from config.database_config import SessionLocal
        
        # Create multiple database sessions
        sessions = []
        
        try:
            for i in range(20):
                session = SessionLocal()
                sessions.append(session)
                
                # Perform a simple query
                from app.models.database import PhoneNumber
                count = session.query(PhoneNumber).count()
                assert count >= 0
            
            # All sessions should be created successfully
            assert len(sessions) == 20
            
        finally:
            # Clean up sessions
            for session in sessions:
                session.close()

    @pytest.mark.asyncio
    async def test_memory_scaling_with_data_size(self, db_session):
        """Test memory usage scaling with data size."""
        from app.models.database import PhoneNumber, FraudReport
        
        process = psutil.Process()
        
        # Test with different data sizes
        data_sizes = [100, 500, 1000]
        memory_usage = []
        
        for size in data_sizes:
            # Clear previous data
            db_session.query(FraudReport).delete()
            db_session.query(PhoneNumber).delete()
            db_session.commit()
            
            # Create test data
            phones = []
            for i in range(size):
                phone = PhoneNumber(number=f"+123456789{i:04d}", country_code="1")
                phones.append(phone)
            
            db_session.bulk_save_objects(phones)
            db_session.flush()
            
            # Create fraud reports
            reports = []
            for phone in phones:
                report = FraudReport(
                    phone_number_id=phone.id,
                    fraud_type="SCAM_CALL",
                    severity="MEDIUM"
                )
                reports.append(report)
            
            db_session.bulk_save_objects(reports)
            db_session.commit()
            
            # Measure memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_usage.append(memory_mb)
        
        # Memory usage should scale reasonably (not exponentially)
        memory_ratio_1_to_2 = memory_usage[1] / memory_usage[0]
        memory_ratio_2_to_3 = memory_usage[2] / memory_usage[1]
        
        # Memory scaling should be roughly linear (ratio < 3x)
        assert memory_ratio_1_to_2 < 3.0
        assert memory_ratio_2_to_3 < 3.0

class TestBottleneckIdentification:
    @pytest.mark.asyncio
    async def test_identify_slow_operations(self, db_session):
        """Identify potentially slow operations."""
        from app.core.fraud_detector import FraudDetector
        from app.models.database import PhoneNumber
        
        # Create test phone
        phone = PhoneNumber(number="+1999999999", country_code="1")
        db_session.add(phone)
        db_session.commit()
        
        detector = FraudDetector(db_session)
        
        # Time different operations
        operations = {}
        
        # Test individual components
        start_time = time.time()
        await detector._calculate_base_risk(phone)
        operations['base_risk'] = time.time() - start_time
        
        start_time = time.time()
        await detector._analyze_network_connections(phone)
        operations['network_analysis'] = time.time() - start_time
        
        start_time = time.time()
        await detector._detect_behavioral_patterns(phone)
        operations['behavior_analysis'] = time.time() - start_time
        
        # Identify the slowest operations
        slowest_op = max(operations.items(), key=lambda x: x[1])
        
        # Log performance insights
        print(f"\nOperation timings:")
        for op, duration in operations.items():
            print(f"  {op}: {duration:.3f}s")
        print(f"Slowest operation: {slowest_op[0]} ({slowest_op[1]:.3f}s)")
        
        # No operation should take more than 1 second for simple case
        assert all(duration < 1.0 for duration in operations.values())

    def test_resource_utilization(self):
        """Test overall resource utilization."""
        process = psutil.Process()
        
        # Get system resources
        cpu_count = psutil.cpu_count()
        memory_total = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
        
        print(f"\nSystem Resources:")
        print(f"  CPU cores: {cpu_count}")
        print(f"  Total memory: {memory_total:.1f} GB")
        print(f"  Current CPU usage: {psutil.cpu_percent()}%")
        print(f"  Current memory usage: {psutil.virtual_memory().percent}%")
        
        # Application resource usage
        app_memory = process.memory_info().rss / 1024 / 1024  # MB
        app_cpu = process.cpu_percent()
        
        print(f"\nApplication Resources:")
        print(f"  Memory usage: {app_memory:.1f} MB")
        print(f"  CPU usage: {app_cpu}%")
        
        # Resource usage should be reasonable
        assert app_memory < 1000  # Less than 1GB
        assert app_cpu < 50  # Less than 50% CPU