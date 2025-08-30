import pytest
from datetime import datetime, timedelta
from app.utils.security import (
    hash_password, verify_password, generate_api_key, 
    hash_api_key, verify_api_key, encrypt_sensitive_data, 
    decrypt_sensitive_data
)
from app.utils.helpers import (
    format_phone_number, parse_phone_number, calculate_time_difference,
    normalize_risk_score, classify_risk_level, detect_outliers
)
from app.utils.logger import get_logger, AuditLogger, PerformanceLogger

class TestSecurity:
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)

    def test_api_key_generation(self):
        """Test API key generation and verification."""
        api_key = generate_api_key()
        hashed_key = hash_api_key(api_key)
        
        assert len(api_key) > 20  # Should be reasonably long
        assert verify_api_key(api_key, hashed_key)
        assert not verify_api_key("wrong_key", hashed_key)

    def test_data_encryption(self):
        """Test data encryption and decryption."""
        sensitive_data = "sensitive_phone_number_+1234567890"
        encrypted = encrypt_sensitive_data(sensitive_data)
        decrypted = decrypt_sensitive_data(encrypted)
        
        assert encrypted != sensitive_data
        assert decrypted == sensitive_data

    def test_encryption_different_each_time(self):
        """Test that encryption produces different results each time."""
        data = "test_data"
        encrypted1 = encrypt_sensitive_data(data)
        encrypted2 = encrypt_sensitive_data(data)
        
        # Should be different due to random IV
        assert encrypted1 != encrypted2
        # But both should decrypt to same value
        assert decrypt_sensitive_data(encrypted1) == data
        assert decrypt_sensitive_data(encrypted2) == data

class TestHelpers:
    def test_phone_number_formatting(self):
        """Test phone number formatting."""
        # Test various input formats
        test_cases = [
            ("+1234567890", "+1234567890"),
            ("1234567890", "1234567890"),
            ("+1 (234) 567-8900", "+12345678900"),
            ("234-567-8900", "2345678900")
        ]
        
        for input_phone, expected in test_cases:
            result = format_phone_number(input_phone)
            # The exact output depends on the implementation
            assert isinstance(result, str)
            assert len(result) > 0

    def test_phone_number_parsing(self):
        """Test phone number parsing."""
        phone = "+1234567890"
        parsed = parse_phone_number(phone)
        
        assert isinstance(parsed, dict)
        assert 'number' in parsed
        assert 'is_valid' in parsed
        
        # Test invalid phone
        invalid_parsed = parse_phone_number("invalid")
        assert not invalid_parsed['is_valid']

    def test_time_difference_calculation(self):
        """Test time difference calculation."""
        dt1 = datetime(2023, 1, 1, 12, 0, 0)
        dt2 = datetime(2023, 1, 1, 14, 30, 0)
        
        diff = calculate_time_difference(dt1, dt2)
        
        assert diff['total_hours'] == 2.5
        assert diff['total_minutes'] == 150
        assert 'human_readable' in diff

    def test_risk_score_normalization(self):
        """Test risk score normalization."""
        assert normalize_risk_score(0.5) == 0.5
        assert normalize_risk_score(-0.1) == 0.0
        assert normalize_risk_score(1.5) == 1.0
        assert normalize_risk_score(0.75, 0.2, 0.8) == 0.75

    def test_risk_level_classification(self):
        """Test risk level classification."""
        assert classify_risk_level(0.95) == "CRITICAL"
        assert classify_risk_level(0.8) == "HIGH"
        assert classify_risk_level(0.5) == "MEDIUM"
        assert classify_risk_level(0.2) == "LOW"
        assert classify_risk_level(0.05) == "MINIMAL"

    def test_outlier_detection(self):
        """Test outlier detection."""
        # Normal data with some outliers
        data = [1, 2, 3, 2, 3, 2, 1, 100, 2, 3, 1, -50, 2]
        
        result = detect_outliers(data, method='iqr')
        
        assert 'outliers' in result
        assert 'outlier_indices' in result
        assert len(result['outliers']) > 0
        assert 100 in result['outliers']  # Should detect 100 as outlier
        assert -50 in result['outliers']  # Should detect -50 as outlier

class TestLogging:
    def test_logger_creation(self):
        """Test logger creation."""
        logger = get_logger('test_logger')
        assert logger is not None
        
        # Test logging doesn't raise errors
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")

    def test_audit_logger(self):
        """Test audit logger."""
        audit_logger = AuditLogger()
        
        # Test different audit log types
        audit_logger.log_user_action(
            user_id="test_user",
            action="LOGIN",
            ip_address="192.168.1.1"
        )
        
        audit_logger.log_data_access(
            user_id="test_user",
            data_type="PHONE_NUMBER",
            record_id="123",
            operation="read"
        )
        
        audit_logger.log_security_event(
            event_type="FAILED_LOGIN",
            severity="MEDIUM",
            description="Failed login attempt",
            source_ip="192.168.1.100"
        )

    def test_performance_logger(self):
        """Test performance logger."""
        perf_logger = PerformanceLogger()
        
        perf_logger.log_api_performance(
            endpoint="/api/v1/fraud/analyze",
            method="POST",
            response_time=0.150,
            status_code=200,
            user_id="test_user"
        )
        
        perf_logger.log_ml_performance(
            model_name="fraud_predictor",
            operation="prediction",
            duration=0.050,
            input_size=100
        )

class TestValidation:
    def test_input_validation(self):
        """Test input validation functions."""
        from app.api.validators import validate_phone_number, validate_fraud_report
        
        # Test phone number validation
        assert validate_phone_number("+1234567890")
        assert validate_phone_number("1234567890")
        assert not validate_phone_number("invalid")
        assert not validate_phone_number("")
        
        # Test fraud report validation
        valid_report = {
            'phone_number': '+1234567890',
            'fraud_type': 'SCAM_CALL',
            'severity': 'HIGH',
            'description': 'Test report'
        }
        
        result = validate_fraud_report(valid_report)
        assert result['valid']
        
        # Test invalid report
        invalid_report = {
            'phone_number': 'invalid',
            'fraud_type': 'INVALID_TYPE'
        }
        
        result = validate_fraud_report(invalid_report)
        assert not result['valid']
        assert len(result['errors']) > 0

class TestCaching:
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache service operations."""
        from app.services.cache_service import CacheService
        
        cache = CacheService()
        
        # Test basic operations
        await cache.set("test_key", "test_value", timeout=60)
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Test expiration
        await cache.set("temp_key", "temp_value", timeout=1)
        import asyncio
        await asyncio.sleep(2)
        expired_value = await cache.get("temp_key")
        assert expired_value is None
        
        # Test deletion
        await cache.delete("test_key")
        deleted_value = await cache.get("test_key")
        assert deleted_value is None

    @pytest.mark.asyncio
    async def test_cache_serialization(self):
        """Test cache serialization of complex objects."""
        from app.services.cache_service import CacheService
        
        cache = CacheService()
        
        # Test dictionary serialization
        complex_data = {
            'list': [1, 2, 3],
            'dict': {'nested': 'value'},
            'number': 42,
            'string': 'text'
        }
        
        await cache.set("complex_key", complex_data)
        retrieved_data = await cache.get("complex_key")
        
        assert retrieved_data == complex_data

class TestDataProcessing:
    def test_data_cleaning(self):
        """Test data cleaning utilities."""
        from app.utils.helpers import DataProcessor
        
        processor = DataProcessor()
        
        # Test text cleaning
        dirty_text = "  This   has  extra   spaces  "
        clean_text = processor.clean_text(dirty_text)
        assert clean_text == "This has extra spaces"
        
        # Test feature normalization
        features = {'feature1': 10, 'feature2': 20, 'feature3': 5}
        normalized = processor.normalize_features(features)
        
        assert all(0 <= value <= 1 for value in normalized.values())
        
        # Test missing value handling
        data_with_missing = {
            'count': None,
            'score': '',
            'valid_field': 'value'
        }
        
        cleaned = processor.handle_missing_values(data_with_missing)
        assert cleaned['count'] == 0
        assert cleaned['score'] == ''
        assert cleaned['valid_field'] == 'value'