import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

# Test configuration
os.environ['TESTING'] = 'true'
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['REDIS_URL'] = 'redis://localhost:6379/15'  # Test database

from main import create_app
from config.database_config import Base, engine, SessionLocal
from app.models.database import *
from app.models.user_models import *
from app.utils.security import hash_password

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def app():
    """Create application for testing."""
    app = create_app('testing')
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "JWT_SECRET_KEY": "test-jwt-secret",
        "SECRET_KEY": "test-secret-key"
    })
    
    with app.app_context():
        yield app

@pytest.fixture(scope="session")
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture(scope="function")
def db_session():
    """Create database session for testing."""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    session = SessionLocal()
    
    yield session
    
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def auth_headers(db_session):
    """Create authentication headers for testing."""
    # Create test user
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash=hash_password("testpass123"),
        is_active=True,
        role="analyst"
    )
    db_session.add(user)
    db_session.commit()
    
    # Create JWT token
    from flask_jwt_extended import create_access_token
    access_token = create_access_token(identity=user.id)
    
    return {"Authorization": f"Bearer {access_token}"}

@pytest.fixture
def admin_headers(db_session):
    """Create admin authentication headers."""
    admin_user = User(
        username="admin",
        email="admin@example.com",
        password_hash=hash_password("adminpass123"),
        is_active=True,
        is_admin=True,
        role="admin"
    )
    db_session.add(admin_user)
    db_session.commit()
    
    from flask_jwt_extended import create_access_token
    access_token = create_access_token(identity=admin_user.id)
    
    return {"Authorization": f"Bearer {access_token}"}

@pytest.fixture
def sample_phone_number():
    """Sample phone number for testing."""
    return "+1234567890"

@pytest.fixture
def sample_phone_record(db_session, sample_phone_number):
    """Create sample phone record."""
    phone = PhoneNumber(
        number=sample_phone_number,
        country_code="1",
        is_mobile=True,
        carrier="Test Carrier"
    )
    db_session.add(phone)
    db_session.commit()
    return phone

@pytest.fixture
def sample_fraud_report(db_session, sample_phone_record):
    """Create sample fraud report."""
    report = FraudReport(
        phone_number_id=sample_phone_record.id,
        fraud_type="SCAM_CALL",
        severity="HIGH",
        description="Test fraud report",
        confidence_score=0.85
    )
    db_session.add(report)
    db_session.commit()
    return report

@pytest.fixture
def mock_external_apis():
    """Mock external API calls."""
    with patch('app.services.external_apis.ExternalAPIService') as mock:
        mock_instance = Mock()
        mock_instance.check_government_fraud_database.return_value = {
            'found': False,
            'risk_score': 0.0
        }
        mock_instance.validate_phone_number.return_value = {
            'valid': True,
            'carrier': 'Test Carrier'
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_cache():
    """Mock cache service."""
    with patch('app.services.cache_service.CacheService') as mock:
        mock_instance = Mock()
        mock_instance.get.return_value = None
        mock_instance.set.return_value = True
        mock.return_value = mock_instance
        yield mock_instance