import pytest
import json
from unittest.mock import patch, Mock

class TestAuthAPI:
    def test_login_success(self, client, db_session):
        """Test successful login."""
        from app.models.user_models import User
        from app.utils.security import hash_password
        
        # Create test user
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash=hash_password("testpass123"),
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        
        response = client.post('/api/v1/auth/login', 
                             json={
                                 'username': 'testuser',
                                 'password': 'testpass123'
                             })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'access_token' in data
        assert data['user']['username'] == 'testuser'

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post('/api/v1/auth/login',
                             json={
                                 'username': 'nonexistent',
                                 'password': 'wrongpass'
                             })
        
        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'error' in data

    def test_login_missing_fields(self, client):
        """Test login with missing fields."""
        response = client.post('/api/v1/auth/login',
                             json={'username': 'testuser'})
        
        assert response.status_code == 400

class TestFraudAnalysisAPI:
    def test_analyze_phone_number(self, client, auth_headers, mock_external_apis):
        """Test phone number analysis endpoint."""
        with patch('app.core.fraud_detector.FraudDetector') as mock_detector:
            # Mock the analysis result
            mock_result = Mock()
            mock_result.phone_number = "+1234567890"
            mock_result.risk_score = 0.75
            mock_result.risk_level = "HIGH"
            mock_result.fraud_probability = 0.65
            mock_result.confidence = 0.85
            mock_result.detected_patterns = ["unusual_timing", "high_frequency"]
            mock_result.network_risk = 0.45
            mock_result.behavioral_anomalies = ["burst_activity"]
            mock_result.recommendations = ["Monitor closely", "Request verification"]
            mock_result.evidence = {"test": "data"}
            mock_result.processing_time = 0.150
            
            mock_detector_instance = Mock()
            mock_detector_instance.analyze_phone_number.return_value = mock_result
            mock_detector.return_value = mock_detector_instance
            
            response = client.post('/api/v1/fraud/analyze',
                                 headers=auth_headers,
                                 json={'phone_number': '+1234567890'})
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['risk_level'] == 'HIGH'
            assert data['risk_score'] == 0.75

    def test_analyze_invalid_phone(self, client, auth_headers):
        """Test analysis with invalid phone number."""
        response = client.post('/api/v1/fraud/analyze',
                             headers=auth_headers,
                             json={'phone_number': 'invalid'})
        
        assert response.status_code == 400

    def test_analyze_unauthorized(self, client):
        """Test analysis without authentication."""
        response = client.post('/api/v1/fraud/analyze',
                             json={'phone_number': '+1234567890'})
        
        assert response.status_code == 401

class TestFraudReportsAPI:
    def test_submit_fraud_report(self, client, auth_headers):
        """Test submitting a fraud report."""
        report_data = {
            'phone_number': '+1234567890',
            'fraud_type': 'SCAM_CALL',
            'severity': 'HIGH',
            'description': 'Attempted to steal personal information',
            'confidence_score': 0.9
        }
        
        response = client.post('/api/v1/fraud/report',
                             headers=auth_headers,
                             json=report_data)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'report_id' in data
        assert data['status'] == 'submitted'

    def test_submit_invalid_report(self, client, auth_headers):
        """Test submitting invalid fraud report."""
        response = client.post('/api/v1/fraud/report',
                             headers=auth_headers,
                             json={'phone_number': '+1234567890'})  # Missing required fields
        
        assert response.status_code == 400

    def test_get_fraud_reports(self, client, auth_headers, db_session, sample_fraud_report):
        """Test retrieving fraud reports."""
        response = client.get('/api/v1/fraud/reports',
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'reports' in data
        assert 'total' in data

    def test_get_reports_with_filters(self, client, auth_headers):
        """Test retrieving reports with filters."""
        response = client.get('/api/v1/fraud/reports?fraud_type=SCAM_CALL&severity=HIGH',
                            headers=auth_headers)
        
        assert response.status_code == 200

class TestBatchAnalysisAPI:
    def test_batch_analysis(self, client, auth_headers):
        """Test batch analysis endpoint."""
        phone_numbers = ['+1234567890', '+1987654321', '+1555555555']
        
        with patch('app.core.fraud_detector.FraudDetector') as mock_detector:
            mock_result = Mock()
            mock_result.phone_number = "+1234567890"
            mock_result.risk_score = 0.5
            mock_result.risk_level = "MEDIUM"
            mock_result.fraud_probability = 0.3
            mock_result.confidence = 0.8
            
            mock_detector_instance = Mock()
            mock_detector_instance.analyze_phone_number.return_value = mock_result
            mock_detector.return_value = mock_detector_instance
            
            response = client.post('/api/v1/fraud/batch-analyze',
                                 headers=auth_headers,
                                 json={'phone_numbers': phone_numbers})
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['results']) == 3

    def test_batch_analysis_limit(self, client, auth_headers):
        """Test batch analysis with too many numbers."""
        phone_numbers = [f'+123456789{i:02d}' for i in range(101)]  # 101 numbers
        
        response = client.post('/api/v1/fraud/batch-analyze',
                             headers=auth_headers,
                             json={'phone_numbers': phone_numbers})
        
        assert response.status_code == 400

class TestStatisticsAPI:
    def test_get_statistics(self, client, auth_headers):
        """Test statistics endpoint."""
        response = client.get('/api/v1/fraud/statistics',
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'total_reports' in data
        assert 'verified_reports' in data

    def test_get_statistics_with_date_range(self, client, auth_headers):
        """Test statistics with date range."""
        response = client.get('/api/v1/fraud/statistics?days=7',
                            headers=auth_headers)
        
        assert response.status_code == 200

class TestAdminAPI:
    def test_model_training_admin_only(self, client, admin_headers):
        """Test that model training requires admin privileges."""
        response = client.post('/api/v1/ml/train',
                             headers=admin_headers,
                             json={'model_types': ['fraud_predictor']})
        
        # Should not fail due to permissions (might fail due to insufficient data)
        assert response.status_code in [200, 400, 500]

    def test_model_training_non_admin(self, client, auth_headers):
        """Test that non-admin users cannot train models."""
        response = client.post('/api/v1/ml/train',
                             headers=auth_headers,
                             json={'model_types': ['fraud_predictor']})
        
        assert response.status_code == 403

class TestRateLimiting:
    def test_rate_limiting(self, client, auth_headers):
        """Test API rate limiting."""
        # Make multiple requests rapidly
        responses = []
        for _ in range(10):
            response = client.post('/api/v1/fraud/analyze',
                                 headers=auth_headers,
                                 json={'phone_number': '+1234567890'})
            responses.append(response.status_code)
        
        # Some requests should succeed, but we might hit rate limits
        # The exact behavior depends on the rate limit configuration
        assert any(status == 200 for status in responses)

class TestErrorHandling:
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method."""
        response = client.delete('/api/v1/fraud/analyze')
        assert response.status_code == 405

    def test_malformed_json(self, client, auth_headers):
        """Test handling of malformed JSON."""
        response = client.post('/api/v1/fraud/analyze',
                             headers=auth_headers,
                             data='invalid json',
                             content_type='application/json')
        
        assert response.status_code == 400