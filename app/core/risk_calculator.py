from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import PhoneNumber, FraudReport, RiskScore, GeolocationData, IdentityLink
from app.utils.logger import get_logger
import math

logger = get_logger(__name__)

class RiskCalculator:
    def __init__(self, db_session: Session):
        self.db = db_session
        
        # Risk factor weights
        self.WEIGHTS = {
            'report_frequency': 0.25,
            'report_severity': 0.20,
            'geographic_anomaly': 0.15,
            'identity_verification': 0.15,
            'temporal_patterns': 0.10,
            'network_associations': 0.10,
            'device_behavior': 0.05
        }
        
        # Severity mapping
        self.SEVERITY_SCORES = {
            'LOW': 0.2,
            'MEDIUM': 0.5,
            'HIGH': 0.8,
            'CRITICAL': 1.0
        }
    
    async def calculate_comprehensive_risk(self, phone_record: PhoneNumber) -> Dict:
        """
        Calculate comprehensive risk score using multiple factors
        """
        try:
            risk_factors = {}
            
            # 1. Report-based risk
            risk_factors.update(await self._calculate_report_risk(phone_record))
            
            # 2. Geographic risk
            risk_factors.update(await self._calculate_geographic_risk(phone_record))
            
            # 3. Identity verification risk
            risk_factors.update(await self._calculate_identity_risk(phone_record))
            
            # 4. Temporal pattern risk
            risk_factors.update(await self._calculate_temporal_risk(phone_record))
            
            # 5. Device behavior risk
            risk_factors.update(await self._calculate_device_risk(phone_record))
            
            # Calculate weighted risk score
            base_risk = self._calculate_weighted_risk(risk_factors)
            
            return {
                'base_risk': base_risk,
                'risk_factors': risk_factors,
                'risk_breakdown': self._generate_risk_breakdown(risk_factors)
            }
            
        except Exception as e:
            logger.error(f"Risk calculation failed for {phone_record.number}: {str(e)}")
            return {'base_risk': 0.0, 'risk_factors': {}}
    
    async def _calculate_report_risk(self, phone_record: PhoneNumber) -> Dict:
        """
        Calculate risk based on fraud reports
        """
        # Get all reports for this phone
        reports = self.db.query(FraudReport).filter_by(
            phone_number_id=phone_record.id
        ).order_by(FraudReport.created_at.desc()).all()
        
        if not reports:
            return {
                'report_frequency_risk': 0.0,
                'report_severity_risk': 0.0,
                'total_reports': 0,
                'verified_reports': 0
            }
        
        # Calculate frequency risk
        now = datetime.now()
        recent_reports = [r for r in reports if r.created_at > now - timedelta(days=30)]
        frequency_risk = min(1.0, len(recent_reports) / 10.0)  # Max risk at 10 reports/month
        
        # Calculate severity risk
        verified_reports = [r for r in reports if r.status == 'VERIFIED']
        if verified_reports:
            avg_severity = sum(self.SEVERITY_SCORES.get(r.severity, 0.5) for r in verified_reports) / len(verified_reports)
            severity_risk = avg_severity
        else:
            severity_risk = 0.0
        
        # Escalation pattern (increasing report frequency)
        escalation_risk = self._calculate_escalation_risk(reports)
        
        return {
            'report_frequency_risk': frequency_risk,
            'report_severity_risk': severity_risk,
            'escalation_risk': escalation_risk,
            'total_reports': len(reports),
            'verified_reports': len(verified_reports),
            'recent_reports': len(recent_reports)
        }
    
    async def _calculate_geographic_risk(self, phone_record: PhoneNumber) -> Dict:
        """
        Calculate risk based on geographic patterns
        """
        # Get location data
        locations = self.db.query(GeolocationData).filter_by(
            phone_number_id=phone_record.id
        ).order_by(GeolocationData.timestamp.desc()).limit(100).all()
        
        if len(locations) < 2:
            return {'geographic_risk': 0.0, 'location_anomalies': []}
        
        anomalies = []
        risk_score = 0.0
        
        # Check for impossible travel speeds
        for i in range(len(locations) - 1):
            loc1, loc2 = locations[i], locations[i + 1]
            if loc1.latitude and loc1.longitude and loc2.latitude and loc2.longitude:
                distance = self._calculate_distance(
                    (loc1.latitude, loc1.longitude),
                    (loc2.latitude, loc2.longitude)
                )
                time_diff = abs((loc1.timestamp - loc2.timestamp).total_seconds() / 3600)  # hours
                
                if time_diff > 0:
                    speed = distance / time_diff  # km/h
                    if speed > 1000:  # Impossible speed (likely spoofing)
                        anomalies.append(f"Impossible travel speed: {speed:.0f} km/h")
                        risk_score = max(risk_score, 0.8)
        
        # Check for high-risk locations
        high_risk_countries = ['XX', 'YY']  # Configure based on fraud data
        for location in locations[:10]:  # Check recent locations
            if location.country in high_risk_countries:
                anomalies.append(f"Activity from high-risk location: {location.country}")
                risk_score = max(risk_score, 0.6)
        
        # Check for location hopping pattern
        unique_countries = set(loc.country for loc in locations[:20] if loc.country)
        if len(unique_countries) > 5:  # Many countries in short time
            anomalies.append("Suspicious location hopping pattern")
            risk_score = max(risk_score, 0.5)
        
        return {
            'geographic_risk': risk_score,
            'location_anomalies': anomalies,
            'unique_locations': len(unique_countries),
            'recent_locations': [loc.country for loc in locations[:5] if loc.country]
        }
    
    async def _calculate_identity_risk(self, phone_record: PhoneNumber) -> Dict:
        """
        Calculate risk based on identity verification status
        """
        # Get identity links
        identity_links = self.db.query(IdentityLink).filter_by(
            phone_number_id=phone_record.id
        ).all()
        
        if not identity_links:
            return {
                'identity_risk': 0.7,  # High risk if no identity verification
                'verification_status': 'NONE',
                'linked_identities': []
            }
        
        risk_score = 0.0
        verified_count = 0
        total_count = len(identity_links)
        
        identity_types = []
        for link in identity_links:
            identity_types.append(link.identity_type)
            if link.verification_status == 'VERIFIED':
                verified_count += 1
            elif link.verification_status == 'REJECTED':
                risk_score = max(risk_score, 0.8)  # Failed verification is suspicious
        
        # Calculate verification ratio
        verification_ratio = verified_count / total_count if total_count > 0 else 0
        
        # Risk decreases with verification
        if verification_ratio >= 0.8:
            risk_score = max(risk_score, 0.1)
        elif verification_ratio >= 0.5:
            risk_score = max(risk_score, 0.3)
        else:
            risk_score = max(risk_score, 0.6)
        
        # Check for suspicious identity patterns
        if len(set(identity_types)) > 3:  # Too many different identity types
            risk_score = max(risk_score, 0.5)
        
        return {
            'identity_risk': risk_score,
            'verification_ratio': verification_ratio,
            'linked_identities': identity_types,
            'verified_count': verified_count
        }
    
    async def _calculate_temporal_risk(self, phone_record: PhoneNumber) -> Dict:
        """
        Calculate risk based on temporal patterns
        """
        # Get reports with timestamps
        reports = self.db.query(FraudReport).filter_by(
            phone_number_id=phone_record.id
        ).order_by(FraudReport.created_at).all()
        
        if len(reports) < 3:
            return {'temporal_risk': 0.0, 'temporal_patterns': []}
        
        patterns = []
        risk_score = 0.0
        
        # Check for clustering in time
        report_times = [r.created_at for r in reports]
        clusters = self._find_time_clusters(report_times)
        
        if len(clusters) > 1:
            patterns.append("Multiple fraud clusters detected")
            risk_score = max(risk_score, 0.6)
        
        # Check for escalating pattern
        recent_reports = [r for r in reports if r.created_at > datetime.now() - timedelta(days=60)]
        if len(recent_reports) > len(reports) * 0.7:  # 70% of reports in last 60 days
            patterns.append("Escalating fraud pattern")
            risk_score = max(risk_score, 0.7)
        
        return {
            'temporal_risk': risk_score,
            'temporal_patterns': patterns,
            'report_clusters': len(clusters)
        }
    
    async def _calculate_device_risk(self, phone_record: PhoneNumber) -> Dict:
        """
        Calculate risk based on device behavior patterns
        """
        # This would analyze device fingerprinting data
        # For now, return placeholder
        return {
            'device_risk': 0.0,
            'device_anomalies': [],
            'device_changes': 0
        }
    
    def _calculate_weighted_risk(self, risk_factors: Dict) -> float:
        """
        Calculate weighted overall risk score
        """
        total_risk = 0.0
        
        # Apply weights to each risk factor
        for factor, weight in self.WEIGHTS.items():
            if factor == 'report_frequency':
                total_risk += risk_factors.get('report_frequency_risk', 0.0) * weight
            elif factor == 'report_severity':
                total_risk += risk_factors.get('report_severity_risk', 0.0) * weight
            elif factor == 'geographic_anomaly':
                total_risk += risk_factors.get('geographic_risk', 0.0) * weight
            elif factor == 'identity_verification':
                total_risk += risk_factors.get('identity_risk', 0.0) * weight
            elif factor == 'temporal_patterns':
                total_risk += risk_factors.get('temporal_risk', 0.0) * weight
            elif factor == 'device_behavior':
                total_risk += risk_factors.get('device_risk', 0.0) * weight
        
        return min(1.0, total_risk)
    
    def _generate_risk_breakdown(self, risk_factors: Dict) -> Dict:
        """
        Generate detailed risk breakdown for reporting
        """
        breakdown = {}
        
        for factor, weight in self.WEIGHTS.items():
            factor_risk = 0.0
            if factor == 'report_frequency':
                factor_risk = risk_factors.get('report_frequency_risk', 0.0)
            elif factor == 'report_severity':
                factor_risk = risk_factors.get('report_severity_risk', 0.0)
            elif factor == 'geographic_anomaly':
                factor_risk = risk_factors.get('geographic_risk', 0.0)
            elif factor == 'identity_verification':
                factor_risk = risk_factors.get('identity_risk', 0.0)
            elif factor == 'temporal_patterns':
                factor_risk = risk_factors.get('temporal_risk', 0.0)
            elif factor == 'device_behavior':
                factor_risk = risk_factors.get('device_risk', 0.0)
            
            breakdown[factor] = {
                'risk_score': factor_risk,
                'weight': weight,
                'contribution': factor_risk * weight
            }
        
        return breakdown
    
    def _calculate_escalation_risk(self, reports: List[FraudReport]) -> float:
        """
        Calculate risk based on escalating report pattern
        """
        if len(reports) < 3:
            return 0.0
        
        # Sort by date
        sorted_reports = sorted(reports, key=lambda x: x.created_at)
        
        # Check if recent reports are more frequent
        now = datetime.now()
        periods = [
            (30, timedelta(days=30)),
            (60, timedelta(days=60)),
            (90, timedelta(days=90))
        ]
        
        frequencies = []
        for days, period in periods:
            count = sum(1 for r in sorted_reports if r.created_at > now - period)
            frequencies.append(count / days * 30)  # Normalize to monthly frequency
        
        # Check if frequency is increasing
        if len(frequencies) >= 2 and frequencies[0] > frequencies[1] * 1.5:
            return 0.8
        elif len(frequencies) >= 3 and frequencies[0] > frequencies[2] * 2:
            return 0.9
        
        return 0.0
    
    def _calculate_distance(self, coord1: tuple, coord2: tuple) -> float:
        """
        Calculate distance between two coordinates using Haversine formula
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    def _find_time_clusters(self, timestamps: List[datetime], threshold_hours: int = 24) -> List[List[datetime]]:
        """
        Find clusters of timestamps within a threshold
        """
        if not timestamps:
            return []
        
        sorted_times = sorted(timestamps)
        clusters = []
        current_cluster = [sorted_times[0]]
        
        for i in range(1, len(sorted_times)):
            time_diff = (sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600
            
            if time_diff <= threshold_hours:
                current_cluster.append(sorted_times[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [sorted_times[i]]
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        return clusters