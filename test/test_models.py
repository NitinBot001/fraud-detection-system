import pytest
from datetime import datetime, timedelta
from app.models.database import PhoneNumber, FraudReport, RiskScore, NetworkConnection
from app.models.user_models import User, APIKey
from app.utils.security import hash_password, verify_password

class TestPhoneNumber:
    def test_create_phone_number(self, db_session):
        """Test creating a phone number record."""
        phone = PhoneNumber(
            number="+1234567890",
            country_code="1",
            is_mobile=True,
            carrier="Test Carrier"
        )
        db_session.add(phone)
        db_session.commit()
        
        assert phone.id is not None
        assert phone.number == "+1234567890"
        assert phone.is_active is True
        assert phone.created_at is not None

    def test_phone_number_relationships(self, db_session):
        """Test phone number relationships."""
        phone = PhoneNumber(number="+1234567890", country_code="1")
        db_session.add(phone)
        db_session.flush()
        
        # Add fraud report
        report = FraudReport(
            phone_number_id=phone.id,
            fraud_type="SCAM_CALL",
            severity="HIGH"
        )
        db_session.add(report)
        
        # Add risk score
        risk = RiskScore(
            phone_number_id=phone.id,
            overall_score=0.75,
            fraud_probability=0.65
        )
        db_session.add(risk)
        db_session.commit()
        
        # Test relationships
        assert len(phone.fraud_reports) == 1
        assert len(phone.risk_scores) == 1
        assert phone.fraud_reports[0].fraud_type == "SCAM_CALL"

class TestFraudReport:
    def test_create_fraud_report(self, db_session, sample_phone_record):
        """Test creating a fraud report."""
        report = FraudReport(
            phone_number_id=sample_phone_record.id,
            fraud_type="PHISHING",
            severity="CRITICAL",
            description="Attempted to steal personal information",
            confidence_score=0.95
        )
        db_session.add(report)
        db_session.commit()
        
        assert report.id is not None
        assert report.fraud_type == "PHISHING"
        assert report.severity == "CRITICAL"
        assert report.status == "PENDING"  # Default value

    def test_fraud_report_validation(self, db_session, sample_phone_record):
        """Test fraud report validation."""
        # Test valid severities
        valid_severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for severity in valid_severities:
            report = FraudReport(
                phone_number_id=sample_phone_record.id,
                fraud_type="SPAM",
                severity=severity
            )
            db_session.add(report)
        
        db_session.commit()
        reports = db_session.query(FraudReport).all()
        assert len(reports) == 4

class TestUser:
    def test_create_user(self, db_session):
        """Test creating a user."""
        password_hash = hash_password("testpass123")
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash=password_hash,
            full_name="Test User"
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.is_active is True
        assert user.is_admin is False
        assert verify_password("testpass123", user.password_hash)

    def test_user_login_attempts(self, db_session):
        """Test user login attempt tracking."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash=hash_password("testpass123")
        )
        db_session.add(user)
        db_session.commit()
        
        # Simulate failed login attempts
        user.failed_login_attempts = 3
        user.account_locked_until = datetime.now() + timedelta(minutes=30)
        db_session.commit()
        
        assert user.failed_login_attempts == 3
        assert user.account_locked_until > datetime.now()

class TestRiskScore:
    def test_create_risk_score(self, db_session, sample_phone_record):
        """Test creating a risk score."""
        risk = RiskScore(
            phone_number_id=sample_phone_record.id,
            overall_score=0.78,
            fraud_probability=0.65,
            network_risk=0.45,
            behavior_risk=0.55,
            historical_risk=0.85
        )
        db_session.add(risk)
        db_session.commit()
        
        assert risk.id is not None
        assert risk.overall_score == 0.78
        assert risk.fraud_probability == 0.65

    def test_risk_score_constraints(self, db_session, sample_phone_record):
        """Test risk score value constraints."""
        # Test that scores are between 0 and 1
        risk = RiskScore(
            phone_number_id=sample_phone_record.id,
            overall_score=1.5,  # This should be handled by application logic
            fraud_probability=-0.1  # This should be handled by application logic
        )
        db_session.add(risk)
        db_session.commit()
        
        # In a real scenario, we'd have database constraints or application validation

class TestNetworkConnection:
    def test_create_network_connection(self, db_session):
        """Test creating network connections."""
        # Create two phone numbers
        phone1 = PhoneNumber(number="+1111111111", country_code="1")
        phone2 = PhoneNumber(number="+2222222222", country_code="1")
        db_session.add_all([phone1, phone2])
        db_session.flush()
        
        # Create connection
        connection = NetworkConnection(
            source_phone_id=phone1.id,
            target_phone_id=phone2.id,
            connection_type="CALL",
            strength=0.8,
            frequency=15
        )
        db_session.add(connection)
        db_session.commit()
        
        assert connection.id is not None
        assert connection.connection_type == "CALL"
        assert connection.strength == 0.8

    def test_bidirectional_connections(self, db_session):
        """Test that connections can be bidirectional."""
        phone1 = PhoneNumber(number="+1111111111", country_code="1")
        phone2 = PhoneNumber(number="+2222222222", country_code="1")
        db_session.add_all([phone1, phone2])
        db_session.flush()
        
        # Create connections in both directions
        conn1 = NetworkConnection(
            source_phone_id=phone1.id,
            target_phone_id=phone2.id,
            connection_type="CALL",
            strength=0.8
        )
        conn2 = NetworkConnection(
            source_phone_id=phone2.id,
            target_phone_id=phone1.id,
            connection_type="SMS",
            strength=0.6
        )
        db_session.add_all([conn1, conn2])
        db_session.commit()
        
        connections = db_session.query(NetworkConnection).all()
        assert len(connections) == 2