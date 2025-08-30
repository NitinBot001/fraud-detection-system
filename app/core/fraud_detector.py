from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
from sqlalchemy.orm import Session
from app.models.database import PhoneNumber, FraudReport, RiskScore, NetworkConnection
from app.core.risk_calculator import RiskCalculator
from app.core.network_analyzer import NetworkAnalyzer
from app.core.pattern_analyzer import PatternAnalyzer
from app.ml.anomaly_detector import AnomalyDetector
from app.ml.fraud_predictor import FraudPredictor
from app.services.cache_service import CacheService
from app.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class FraudDetectionResult:
    phone_number: str
    risk_score: float
    fraud_probability: float
    risk_level: str
    confidence: float
    detected_patterns: List[str]
    network_risk: float
    behavioral_anomalies: List[str]
    recommendations: List[str]
    evidence: Dict
    processing_time: float

class FraudDetector:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.risk_calculator = RiskCalculator(db_session)
        self.network_analyzer = NetworkAnalyzer(db_session)
        self.pattern_analyzer = PatternAnalyzer(db_session)
        self.anomaly_detector = AnomalyDetector()
        self.fraud_predictor = FraudPredictor()
        self.cache = CacheService()
        
        # Risk thresholds
        self.RISK_THRESHOLDS = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'CRITICAL': 0.95
        }
    
    async def analyze_phone_number(self, phone_number: str, deep_analysis: bool = True) -> FraudDetectionResult:
        """
        Comprehensive fraud analysis for a phone number
        """
        start_time = datetime.now()
        
        try:
            # Check cache first for recent analysis
            cache_key = f"fraud_analysis:{phone_number}"
            cached_result = await self.cache.get(cache_key)
            if cached_result and not deep_analysis:
                return cached_result
            
            # Get or create phone number record
            phone_record = self._get_or_create_phone_record(phone_number)
            
            # Parallel analysis tasks
            analysis_tasks = [
                self._calculate_base_risk(phone_record),
                self._analyze_network_connections(phone_record),
                self._detect_behavioral_patterns(phone_record),
                self._check_historical_reports(phone_record),
                self._analyze_geolocation_patterns(phone_record)
            ]
            
            if deep_analysis:
                analysis_tasks.extend([
                    self._run_ml_anomaly_detection(phone_record),
                    self._predict_fraud_probability(phone_record),
                    self._analyze_identity_links(phone_record)
                ])
            
            # Execute analysis tasks concurrently
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Compile results
            fraud_result = self._compile_analysis_results(
                phone_number, phone_record, results, start_time
            )
            
            # Update risk score in database
            await self._update_risk_score(phone_record, fraud_result)
            
            # Cache result
            await self.cache.set(cache_key, fraud_result, timeout=300)
            
            logger.info(f"Fraud analysis completed for {phone_number}", extra={
                'phone_number': phone_number,
                'risk_score': fraud_result.risk_score,
                'processing_time': fraud_result.processing_time
            })
            
            return fraud_result
            
        except Exception as e:
            logger.error(f"Fraud analysis failed for {phone_number}: {str(e)}")
            raise
    
    def _get_or_create_phone_record(self, phone_number: str) -> PhoneNumber:
        """Get existing phone record or create new one"""
        phone_record = self.db.query(PhoneNumber).filter_by(number=phone_number).first()
        
        if not phone_record:
            # Parse phone number for metadata
            import phonenumbers
            try:
                parsed = phonenumbers.parse(phone_number, None)
                country_code = str(parsed.country_code)
                is_mobile = phonenumbers.number_type(parsed) in [
                    phonenumbers.PhoneNumberType.MOBILE,
                    phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE
                ]
            except:
                country_code = "unknown"
                is_mobile = True
            
            phone_record = PhoneNumber(
                number=phone_number,
                country_code=country_code,
                is_mobile=is_mobile
            )
            self.db.add(phone_record)
            self.db.commit()
        
        return phone_record
    
    async def _calculate_base_risk(self, phone_record: PhoneNumber) -> Dict:
        """Calculate base risk score using multiple factors"""
        return await self.risk_calculator.calculate_comprehensive_risk(phone_record)
    
    async def _analyze_network_connections(self, phone_record: PhoneNumber) -> Dict:
        """Analyze network connections and associations"""
        return await self.network_analyzer.analyze_phone_network(phone_record)
    
    async def _detect_behavioral_patterns(self, phone_record: PhoneNumber) -> Dict:
        """Detect and analyze behavioral patterns"""
        return await self.pattern_analyzer.analyze_behavior_patterns(phone_record)
    
    async def _check_historical_reports(self, phone_record: PhoneNumber) -> Dict:
        """Check historical fraud reports"""
        reports = self.db.query(FraudReport).filter_by(
            phone_number_id=phone_record.id
        ).order_by(FraudReport.created_at.desc()).limit(100).all()
        
        if not reports:
            return {'historical_risk': 0.0, 'report_count': 0, 'recent_reports': 0}
        
        # Calculate historical risk based on reports
        recent_reports = sum(1 for r in reports if r.created_at > datetime.now() - timedelta(days=30))
        severity_weights = {'LOW': 0.2, 'MEDIUM': 0.5, 'HIGH': 0.8, 'CRITICAL': 1.0}
        
        historical_risk = min(1.0, sum(severity_weights.get(r.severity, 0.5) for r in reports) / 10)
        
        return {
            'historical_risk': historical_risk,
            'report_count': len(reports),
            'recent_reports': recent_reports,
            'fraud_types': list(set(r.fraud_type for r in reports))
        }
    
    async def _analyze_geolocation_patterns(self, phone_record: PhoneNumber) -> Dict:
        """Analyze geolocation patterns for anomalies"""
        # Implementation would analyze location data for suspicious patterns
        # This is a placeholder for the complex geolocation analysis
        return {
            'geographic_risk': 0.0,
            'location_anomalies': [],
            'travel_patterns': []
        }
    
    async def _run_ml_anomaly_detection(self, phone_record: PhoneNumber) -> Dict:
        """Run ML-based anomaly detection"""
        try:
            features = await self._extract_ml_features(phone_record)
            anomaly_score = await self.anomaly_detector.detect_anomalies(features)
            
            return {
                'anomaly_score': anomaly_score,
                'is_anomaly': anomaly_score > 0.7,
                'anomaly_features': features
            }
        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {str(e)}")
            return {'anomaly_score': 0.0, 'is_anomaly': False}
    
    async def _predict_fraud_probability(self, phone_record: PhoneNumber) -> Dict:
        """Predict fraud probability using ML models"""
        try:
            features = await self._extract_ml_features(phone_record)
            fraud_prob = await self.fraud_predictor.predict_fraud_probability(features)
            
            return {
                'fraud_probability': fraud_prob,
                'prediction_confidence': 0.85,  # Model confidence
                'model_version': '1.0'
            }
        except Exception as e:
            logger.warning(f"Fraud prediction failed: {str(e)}")
            return {'fraud_probability': 0.0, 'prediction_confidence': 0.0}
    
    async def _analyze_identity_links(self, phone_record: PhoneNumber) -> Dict:
        """Analyze identity linkages"""
        # Implementation would analyze linked identities (Aadhar, email, etc.)
        return {
            'identity_risk': 0.0,
            'linked_identities': [],
            'suspicious_links': []
        }
    
    async def _extract_ml_features(self, phone_record: PhoneNumber) -> Dict:
        """Extract features for ML models"""
        # This would extract comprehensive features for ML analysis
        # Including call patterns, network features, historical data, etc.
        return {
            'phone_age': 30,  # days since first seen
            'report_count': 0,
            'network_size': 0,
            'activity_pattern': [],
            # ... many more features
        }
    
    def _compile_analysis_results(self, phone_number: str, phone_record: PhoneNumber, 
                                results: List, start_time: datetime) -> FraudDetectionResult:
        """Compile all analysis results into final fraud detection result"""
        
        # Extract results (handling exceptions)
        base_risk = results[0] if not isinstance(results[0], Exception) else {}
        network_analysis = results[1] if not isinstance(results[1], Exception) else {}
        behavioral_analysis = results[2] if not isinstance(results[2], Exception) else {}
        historical_analysis = results[3] if not isinstance(results[3], Exception) else {}
        geo_analysis = results[4] if not isinstance(results[4], Exception) else {}
        
        # Calculate overall risk score
        risk_components = {
            'base_risk': base_risk.get('base_risk', 0.0),
            'network_risk': network_analysis.get('network_risk', 0.0),
            'behavioral_risk': behavioral_analysis.get('behavioral_risk', 0.0),
            'historical_risk': historical_analysis.get('historical_risk', 0.0),
            'geographic_risk': geo_analysis.get('geographic_risk', 0.0)
        }
        
        # Weighted average of risk components
        weights = {'base_risk': 0.2, 'network_risk': 0.25, 'behavioral_risk': 0.25, 
                  'historical_risk': 0.2, 'geographic_risk': 0.1}
        
        overall_risk = sum(risk_components[k] * weights[k] for k in weights)
        
        # Determine risk level
        risk_level = 'LOW'
        for level, threshold in sorted(self.RISK_THRESHOLDS.items(), 
                                     key=lambda x: x[1], reverse=True):
            if overall_risk >= threshold:
                risk_level = level
                break
        
        # Compile detected patterns
        detected_patterns = []
        detected_patterns.extend(behavioral_analysis.get('detected_patterns', []))
        detected_patterns.extend(network_analysis.get('suspicious_patterns', []))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_risk, detected_patterns)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FraudDetectionResult(
            phone_number=phone_number,
            risk_score=overall_risk,
            fraud_probability=historical_analysis.get('historical_risk', 0.0),
            risk_level=risk_level,
            confidence=0.85,  # Calculate based on data quality
            detected_patterns=detected_patterns,
            network_risk=risk_components['network_risk'],
            behavioral_anomalies=behavioral_analysis.get('anomalies', []),
            recommendations=recommendations,
            evidence={
                'risk_components': risk_components,
                'analysis_details': {
                    'base_risk': base_risk,
                    'network_analysis': network_analysis,
                    'behavioral_analysis': behavioral_analysis,
                    'historical_analysis': historical_analysis
                }
            },
            processing_time=processing_time
        )
    
    def _generate_recommendations(self, risk_score: float, patterns: List[str]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.append("IMMEDIATE ACTION: Block number and investigate")
            recommendations.append("Notify relevant authorities")
        elif risk_score > 0.6:
            recommendations.append("Monitor closely for suspicious activity")
            recommendations.append("Request additional verification")
        elif risk_score > 0.3:
            recommendations.append("Add to watch list")
            recommendations.append("Increase monitoring frequency")
        else:
            recommendations.append("Continue normal monitoring")
        
        return recommendations
    
    async def _update_risk_score(self, phone_record: PhoneNumber, 
                               fraud_result: FraudDetectionResult):
        """Update risk score in database"""
        try:
            # Check if risk score record exists
            risk_score = self.db.query(RiskScore).filter_by(
                phone_number_id=phone_record.id
            ).first()
            
            if risk_score:
                # Update existing record
                risk_score.overall_score = fraud_result.risk_score
                risk_score.fraud_probability = fraud_result.fraud_probability
                risk_score.network_risk = fraud_result.network_risk
                risk_score.prediction_confidence = fraud_result.confidence
            else:
                # Create new record
                risk_score = RiskScore(
                    phone_number_id=phone_record.id,
                    overall_score=fraud_result.risk_score,
                    fraud_probability=fraud_result.fraud_probability,
                    network_risk=fraud_result.network_risk,
                    prediction_confidence=fraud_result.confidence
                )
                self.db.add(risk_score)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update risk score: {str(e)}")
            self.db.rollback()