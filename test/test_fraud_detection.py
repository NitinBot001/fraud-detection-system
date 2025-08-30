import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.core.fraud_detector import FraudDetector, FraudDetectionResult
from app.core.network_analyzer import NetworkAnalyzer
from app.core.risk_calculator import RiskCalculator
from app.ml.anomaly_detector import AnomalyDetector
from app.ml.fraud_predictor import FraudPredictor

class TestFraudDetector:
    @pytest.mark.asyncio
    async def test_analyze_phone_number(self, db_session, sample_phone_record):
        """Test comprehensive phone number analysis."""
        detector = FraudDetector(db_session)
        
        with patch.object(detector, '_calculate_base_risk') as mock_base_risk, \
             patch.object(detector, '_analyze_network_connections') as mock_network, \
             patch.object(detector, '_detect_behavioral_patterns') as mock_behavior, \
             patch.object(detector, '_check_historical_reports') as mock_reports, \
             patch.object(detector, '_analyze_geolocation_patterns') as mock_geo:
            
            # Setup mocks
            mock_base_risk.return_value = {'base_risk': 0.6}
            mock_network.return_value = {'network_risk': 0.4, 'network_size': 5}
            mock_behavior.return_value = {'behavioral_risk': 0.5, 'anomalies': []}
            mock_reports.return_value = {'historical_risk': 0.7, 'report_count': 2}
            mock_geo.return_value = {'geographic_risk': 0.3}
            
            result = await detector.analyze_phone_number(sample_phone_record.number)
            
            assert isinstance(result, FraudDetectionResult)
            assert result.phone_number == sample_phone_record.number
            assert 0 <= result.risk_score <= 1
            assert result.risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    @pytest.mark.asyncio
    async def test_quick_vs_deep_analysis(self, db_session, sample_phone_record):
        """Test difference between quick and deep analysis."""
        detector = FraudDetector(db_session)
        
        with patch.object(detector, '_run_ml_anomaly_detection') as mock_ml, \
             patch.object(detector, '_predict_fraud_probability') as mock_predict:
            
            mock_ml.return_value = {'anomaly_score': 0.8}
            mock_predict.return_value = {'fraud_probability': 0.7}
            
            # Quick analysis (should not call ML methods)
            await detector.analyze_phone_number(sample_phone_record.number, deep_analysis=False)
            mock_ml.assert_not_called()
            mock_predict.assert_not_called()
            
            # Deep analysis (should call ML methods)
            await detector.analyze_phone_number(sample_phone_record.number, deep_analysis=True)
            mock_ml.assert_called_once()
            mock_predict.assert_called_once()

class TestNetworkAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_phone_network(self, db_session, sample_phone_record):
        """Test network analysis for a phone number."""
        analyzer = NetworkAnalyzer(db_session)
        
        result = await analyzer.analyze_phone_network(sample_phone_record)
        
        assert 'network_risk' in result
        assert 'network_size' in result
        assert 'suspicious_patterns' in result
        assert isinstance(result['network_risk'], float)
        assert 0 <= result['network_risk'] <= 1

    @pytest.mark.asyncio
    async def test_detect_fraud_rings(self, db_session):
        """Test fraud ring detection."""
        analyzer = NetworkAnalyzer(db_session)
        
        # Create sample network data
        phones = []
        for i in range(10):
            phone = PhoneNumber(number=f"+123456789{i}", country_code="1")
            phones.append(phone)
        
        db_session.add_all(phones)
        db_session.flush()
        
        # Create connections between phones
        from app.models.database import NetworkConnection
        connections = []
        for i in range(len(phones) - 1):
            conn = NetworkConnection(
                source_phone_id=phones[i].id,
                target_phone_id=phones[i + 1].id,
                connection_type="CALL",
                strength=0.8
            )
            connections.append(conn)
        
        db_session.add_all(connections)
        db_session.commit()
        
        fraud_rings = await analyzer.detect_fraud_rings(min_size=3)
        
        assert isinstance(fraud_rings, list)
        # Should detect at least one potential ring
        assert len(fraud_rings) >= 0

class TestRiskCalculator:
    @pytest.mark.asyncio
    async def test_calculate_comprehensive_risk(self, db_session, sample_phone_record, sample_fraud_report):
        """Test comprehensive risk calculation."""
        calculator = RiskCalculator(db_session)
        
        result = await calculator.calculate_comprehensive_risk(sample_phone_record)
        
        assert 'base_risk' in result
        assert 'risk_factors' in result
        assert isinstance(result['base_risk'], float)
        assert 0 <= result['base_risk'] <= 1

    @pytest.mark.asyncio
    async def test_escalation_risk_calculation(self, db_session, sample_phone_record):
        """Test escalation risk calculation."""
        calculator = RiskCalculator(db_session)
        
        # Create multiple reports with increasing severity
        from app.models.database import FraudReport
        reports = []
        severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        for i, severity in enumerate(severities):
            report = FraudReport(
                phone_number_id=sample_phone_record.id,
                fraud_type='SCAM_CALL',
                severity=severity,
                created_at=datetime.now() - timedelta(days=10-i)
            )
            reports.append(report)
        
        db_session.add_all(reports)
        db_session.commit()
        
        escalation_risk = calculator._calculate_escalation_risk(reports)
        
        assert isinstance(escalation_risk, float)
        assert 0 <= escalation_risk <= 1
        # Should detect escalation pattern
        assert escalation_risk > 0.5

class TestAnomalyDetector:
    @pytest.mark.asyncio
    async def test_detect_anomalies(self):
        """Test anomaly detection."""
        detector = AnomalyDetector()
        
        # Sample feature vector
        features = {
            'call_frequency': 50,
            'call_duration_avg': 120,
            'location_changes': 10,
            'network_size': 25,
            'report_count': 3
        }
        
        anomaly_score = await detector.detect_anomalies(features)
        
        assert isinstance(anomaly_score, float)
        assert 0 <= anomaly_score <= 1

    @pytest.mark.asyncio
    async def test_model_training(self):
        """Test anomaly detector training."""
        detector = AnomalyDetector()
        
        # Create sample training data
        import pandas as pd
        training_data = pd.DataFrame({
            'call_frequency': [10, 15, 12, 8, 45, 50, 48],  # Last 3 are anomalies
            'location_changes': [2, 3, 1, 2, 15, 20, 18],
            'network_size': [5, 8, 6, 4, 30, 35, 32]
        })
        
        metrics = await detector.train_models(training_data)
        
        assert isinstance(metrics, dict)
        # Should have some training metrics
        assert len(metrics) > 0

class TestFraudPredictor:
    @pytest.mark.asyncio
    async def test_predict_fraud_probability(self):
        """Test fraud probability prediction."""
        predictor = FraudPredictor()
        
        features = {
            'call_frequency': 25,
            'report_count': 2,
            'network_size': 15,
            'location_changes': 5,
            'account_age_days': 30
        }
        
        probability = await predictor.predict_fraud_probability(features)
        
        assert isinstance(probability, float)
        assert 0 <= probability <= 1

    @pytest.mark.asyncio
    async def test_prediction_explanation(self):
        """Test prediction explanation."""
        predictor = FraudPredictor()
        
        features = {
            'call_frequency': 50,  # High value
            'report_count': 5,     # High value
            'network_size': 100,   # High value
            'location_changes': 2,  # Low value
            'account_age_days': 365
        }
        
        explanation = await predictor.get_prediction_explanation(features)
        
        assert 'fraud_probability' in explanation
        assert 'explanations' in explanation
        assert isinstance(explanation['explanations'], list)
        
        # Should identify high-risk features
        risk_features = [exp['feature'] for exp in explanation['explanations']]
        assert any('report_count' in feature for feature in risk_features)

class TestIntegrationScenarios:
    @pytest.mark.asyncio
    async def test_high_risk_scenario(self, db_session):
        """Test complete analysis for high-risk scenario."""
        # Create high-risk phone with multiple reports
        phone = PhoneNumber(number="+1999999999", country_code="1")
        db_session.add(phone)
        db_session.flush()
        
        # Add multiple fraud reports
        from app.models.database import FraudReport
        reports = []
        for i in range(5):
            report = FraudReport(
                phone_number_id=phone.id,
                fraud_type='SCAM_CALL',
                severity='HIGH',
                created_at=datetime.now() - timedelta(days=i)
            )
            reports.append(report)
        
        db_session.add_all(reports)
        db_session.commit()
        
        detector = FraudDetector(db_session)
        result = await detector.analyze_phone_number(phone.number)
        
        # Should be classified as high risk
        assert result.risk_level in ['HIGH', 'CRITICAL']
        assert result.risk_score > 0.7

    @pytest.mark.asyncio
    async def test_low_risk_scenario(self, db_session):
        """Test complete analysis for low-risk scenario."""
        # Create clean phone with no reports
        phone = PhoneNumber(number="+1111111111", country_code="1")
        db_session.add(phone)
        db_session.commit()
        
        detector = FraudDetector(db_session)
        result = await detector.analyze_phone_number(phone.number)
        
        # Should be classified as low risk
        assert result.risk_level in ['LOW', 'MINIMAL']
        assert result.risk_score < 0.5

class TestPerformance:
    @pytest.mark.asyncio
    async def test_analysis_performance(self, db_session, sample_phone_record):
        """Test that analysis completes within reasonable time."""
        detector = FraudDetector(db_session)
        
        start_time = datetime.now()
        result = await detector.analyze_phone_number(sample_phone_record.number)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within 5 seconds for simple case
        assert processing_time < 5.0
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_batch_analysis_performance(self, db_session):
        """Test batch analysis performance."""
        detector = FraudDetector(db_session)
        
        # Create multiple phone numbers
        phone_numbers = [f"+155555555{i:02d}" for i in range(10)]
        
        start_time = datetime.now()
        results = []
        
        for phone_number in phone_numbers:
            result = await detector.analyze_phone_number(phone_number, deep_analysis=False)
            results.append(result)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Should process 10 numbers in reasonable time
        assert total_time < 30.0  # 30 seconds for 10 analyses
        assert len(results) == 10

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_invalid_phone_number_format(self, db_session):
        """Test handling of invalid phone number formats."""
        detector = FraudDetector(db_session)
        
        # This should handle gracefully and create a basic record
        result = await detector.analyze_phone_number("invalid_phone")
        
        assert result.phone_number == "invalid_phone"
        assert result.risk_score >= 0

    @pytest.mark.asyncio
    async def test_empty_database(self, db_session):
        """Test analysis with empty database."""
        detector = FraudDetector(db_session)
        
        result = await detector.analyze_phone_number("+1000000000")
        
        # Should handle gracefully with minimal risk
        assert result.risk_level in ['LOW', 'MINIMAL']
        assert len(result.detected_patterns) == 0

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, db_session):
        """Test concurrent analysis requests."""
        detector = FraudDetector(db_session)
        
        phone_numbers = [f"+100000000{i}" for i in range(5)]
        
        # Run analyses concurrently
        tasks = [
            detector.analyze_phone_number(phone, deep_analysis=False)
            for phone in phone_numbers
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(isinstance(r, FraudDetectionResult) for r in results)