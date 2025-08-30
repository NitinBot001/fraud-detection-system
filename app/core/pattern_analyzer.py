from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import PhoneNumber, FraudReport, BehaviorPattern, GeolocationData
from app.utils.logger import get_logger
import json
import numpy as np
from collections import defaultdict, Counter

logger = get_logger(__name__)

class PatternAnalyzer:
    def __init__(self, db_session: Session):
        self.db = db_session
        
        # Pattern detection thresholds
        self.ANOMALY_THRESHOLD = 0.7
        self.MIN_PATTERN_SAMPLES = 5
        
        # Known fraud patterns
        self.FRAUD_PATTERNS = {
            'time_clustering': {'weight': 0.8, 'description': 'Reports clustered in time'},
            'location_spoofing': {'weight': 0.9, 'description': 'Impossible location changes'},
            'burst_activity': {'weight': 0.7, 'description': 'Sudden burst of activity'},
            'identity_cycling': {'weight': 0.8, 'description': 'Rapid identity changes'},
            'network_mirroring': {'weight': 0.6, 'description': 'Mirroring network patterns'}
        }
    
    async def analyze_behavior_patterns(self, phone_record: PhoneNumber) -> Dict:
        """
        Comprehensive behavioral pattern analysis
        """
        try:
            # Get historical behavior data
            patterns = await self._extract_behavior_patterns(phone_record)
            
            # Detect anomalies in patterns
            anomalies = await self._detect_pattern_anomalies(phone_record, patterns)
            
            # Analyze temporal patterns
            temporal_analysis = await self._analyze_temporal_patterns(phone_record)
            
            # Analyze communication patterns
            communication_analysis = await self._analyze_communication_patterns(phone_record)
            
            # Calculate overall behavioral risk
            behavioral_risk = self._calculate_behavioral_risk(anomalies, temporal_analysis, communication_analysis)
            
            return {
                'behavioral_risk': behavioral_risk,
                'detected_patterns': list(patterns.keys()),
                'anomalies': anomalies,
                'temporal_patterns': temporal_analysis,
                'communication_patterns': communication_analysis,
                'pattern_confidence': self._calculate_pattern_confidence(patterns)
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis failed for {phone_record.number}: {str(e)}")
            return {'behavioral_risk': 0.0, 'anomalies': []}
    
    async def _extract_behavior_patterns(self, phone_record: PhoneNumber) -> Dict:
        """
        Extract various behavioral patterns from historical data
        """
        patterns = {}
        
        # Get fraud reports for pattern analysis
        reports = self.db.query(FraudReport).filter_by(
            phone_number_id=phone_record.id
        ).order_by(FraudReport.created_at).all()
        
        if reports:
            # Time-based patterns
            patterns['temporal'] = self._extract_temporal_patterns(reports)
            
            # Frequency patterns
            patterns['frequency'] = self._extract_frequency_patterns(reports)
            
            # Severity progression patterns
            patterns['severity'] = self._extract_severity_patterns(reports)
        
        # Get location data for geographic patterns
        locations = self.db.query(GeolocationData).filter_by(
            phone_number_id=phone_record.id
        ).order_by(GeolocationData.timestamp).all()
        
        if locations:
            patterns['geographic'] = self._extract_geographic_patterns(locations)
        
        # Get existing behavior patterns from database
        stored_patterns = self.db.query(BehaviorPattern).filter_by(
            phone_number_id=phone_record.id
        ).all()
        
        for pattern in stored_patterns:
            patterns[pattern.pattern_type] = json.loads(pattern.pattern_data)
        
        return patterns
    
    async def _detect_pattern_anomalies(self, phone_record: PhoneNumber, patterns: Dict) -> List[str]:
        """
        Detect anomalies in behavioral patterns
        """
        anomalies = []
        
        # Check temporal anomalies
        if 'temporal' in patterns:
            temporal_anomalies = self._detect_temporal_anomalies(patterns['temporal'])
            anomalies.extend(temporal_anomalies)
        
        # Check frequency anomalies
        if 'frequency' in patterns:
            frequency_anomalies = self._detect_frequency_anomalies(patterns['frequency'])
            anomalies.extend(frequency_anomalies)
        
        # Check geographic anomalies
        if 'geographic' in patterns:
            geo_anomalies = self._detect_geographic_anomalies(patterns['geographic'])
            anomalies.extend(geo_anomalies)
        
        # Check for pattern breaks (sudden changes in established patterns)
        pattern_breaks = self._detect_pattern_breaks(patterns)
        anomalies.extend(pattern_breaks)
        
        return anomalies
    
    async def _analyze_temporal_patterns(self, phone_record: PhoneNumber) -> Dict:
        """
        Analyze temporal patterns in phone activity
        """
        reports = self.db.query(FraudReport).filter_by(
            phone_number_id=phone_record.id
        ).order_by(FraudReport.created_at).all()
        
        if len(reports) < 3:
            return {'risk_score': 0.0, 'patterns': []}
        
        # Extract hour-of-day patterns
        hours = [r.created_at.hour for r in reports]
        hour_pattern = Counter(hours)
        
        # Extract day-of-week patterns
        days = [r.created_at.weekday() for r in reports]
        day_pattern = Counter(days)
        
        # Detect unusual timing patterns
        patterns = []
        risk_score = 0.0
        
        # Check for off-hours activity (very early morning)
        off_hours_activity = sum(hour_pattern.get(h, 0) for h in range(2, 6))
        if off_hours_activity > len(reports) * 0.3:
            patterns.append("High off-hours activity")
            risk_score = max(risk_score, 0.6)
        
        # Check for weekend concentration
        weekend_activity = hour_pattern.get(5, 0) + hour_pattern.get(6, 0)  # Fri, Sat
        if weekend_activity > len(reports) * 0.5:
            patterns.append("Weekend activity concentration")
            risk_score = max(risk_score, 0.4)
        
        # Check for burst patterns
        report_intervals = []
        for i in range(1, len(reports)):
            interval = (reports[i].created_at - reports[i-1].created_at).total_seconds() / 3600
            report_intervals.append(interval)
        
        if report_intervals:
            avg_interval = np.mean(report_intervals)
            short_intervals = sum(1 for interval in report_intervals if interval < avg_interval * 0.1)
            
            if short_intervals > len(report_intervals) * 0.3:
                patterns.append("Burst activity pattern detected")
                risk_score = max(risk_score, 0.7)
        
        return {
            'risk_score': risk_score,
            'patterns': patterns,
            'hour_distribution': dict(hour_pattern),
            'day_distribution': dict(day_pattern),
            'avg_interval_hours': np.mean(report_intervals) if report_intervals else 0
        }
    
    async def _analyze_communication_patterns(self, phone_record: PhoneNumber) -> Dict:
        """
        Analyze communication-related patterns
        """
        # This would analyze call logs, SMS patterns, etc.
        # For now, return placeholder analysis
        return {
            'risk_score': 0.0,
            'patterns': [],
            'call_frequency': 0,
            'sms_frequency': 0,
            'contact_diversity': 0
        }
    
    def _extract_temporal_patterns(self, reports: List[FraudReport]) -> Dict:
        """
        Extract temporal patterns from reports
        """
        if not reports:
            return {}
        
        timestamps = [r.created_at for r in reports]
        
        return {
            'first_report': min(timestamps),
            'last_report': max(timestamps),
            'total_span_days': (max(timestamps) - min(timestamps)).days,
            'average_frequency': len(reports) / max(1, (max(timestamps) - min(timestamps)).days),
            'hour_distribution': Counter(t.hour for t in timestamps),
            'day_distribution': Counter(t.weekday() for t in timestamps)
        }
    
    def _extract_frequency_patterns(self, reports: List[FraudReport]) -> Dict:
        """
        Extract frequency patterns from reports
        """
        if len(reports) < 2:
            return {}
        
        # Calculate intervals between reports
        intervals = []
        sorted_reports = sorted(reports, key=lambda x: x.created_at)
        
        for i in range(1, len(sorted_reports)):
            interval = (sorted_reports[i].created_at - sorted_reports[i-1].created_at).total_seconds() / 3600
            intervals.append(interval)
        
        return {
            'total_reports': len(reports),
            'average_interval_hours': np.mean(intervals) if intervals else 0,
            'median_interval_hours': np.median(intervals) if intervals else 0,
            'min_interval_hours': min(intervals) if intervals else 0,
            'max_interval_hours': max(intervals) if intervals else 0,
            'interval_variance': np.var(intervals) if intervals else 0
        }
    
    def _extract_severity_patterns(self, reports: List[FraudReport]) -> Dict:
        """
        Extract severity progression patterns
        """
        if not reports:
            return {}
        
        sorted_reports = sorted(reports, key=lambda x: x.created_at)
        severities = [r.severity for r in sorted_reports]
        
        severity_scores = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        numeric_severities = [severity_scores.get(s, 2) for s in severities]
        
        # Check for escalation pattern
        is_escalating = False
        if len(numeric_severities) >= 3:
            trend = np.polyfit(range(len(numeric_severities)), numeric_severities, 1)[0]
            is_escalating = trend > 0.1
        
        return {
            'severity_distribution': Counter(severities),
            'is_escalating': is_escalating,
            'average_severity': np.mean(numeric_severities) if numeric_severities else 0,
            'max_severity': max(severities) if severities else 'LOW',
            'recent_severity_trend': numeric_severities[-3:] if len(numeric_severities) >= 3 else numeric_severities
        }
    
    def _extract_geographic_patterns(self, locations: List[GeolocationData]) -> Dict:
        """
        Extract geographic movement patterns
        """
        if not locations:
            return {}
        
        # Calculate movement patterns
        movements = []
        for i in range(1, len(locations)):
            if (locations[i].latitude and locations[i].longitude and 
                locations[i-1].latitude and locations[i-1].longitude):
                
                distance = self._calculate_distance(
                    (locations[i-1].latitude, locations[i-1].longitude),
                    (locations[i].latitude, locations[i].longitude)
                )
                time_diff = (locations[i].timestamp - locations[i-1].timestamp).total_seconds() / 3600
                
                movements.append({
                    'distance_km': distance,
                    'time_hours': time_diff,
                    'speed_kmh': distance / max(time_diff, 0.01)
                })
        
        if not movements:
            return {'total_locations': len(locations)}
        
        speeds = [m['speed_kmh'] for m in movements]
        distances = [m['distance_km'] for m in movements]
        
        return {
            'total_locations': len(locations),
            'unique_countries': len(set(loc.country for loc in locations if loc.country)),
            'average_speed': np.mean(speeds),
            'max_speed': max(speeds),
            'total_distance': sum(distances),
            'suspicious_speeds': sum(1 for s in speeds if s > 1000),  # Impossible speeds
            'location_changes': len(movements)
        }
    
    def _detect_temporal_anomalies(self, temporal_patterns: Dict) -> List[str]:
        """
        Detect temporal anomalies
        """
        anomalies = []
        
        # Check for unusually high frequency
        avg_frequency = temporal_patterns.get('average_frequency', 0)
        if avg_frequency > 5:  # More than 5 reports per day on average
            anomalies.append("Unusually high report frequency")
        
        # Check for off-hours concentration
        hour_dist = temporal_patterns.get('hour_distribution', {})
        off_hours = sum(hour_dist.get(h, 0) for h in range(0, 6))
        total_reports = sum(hour_dist.values()) if hour_dist else 1
        
        if off_hours / total_reports > 0.5:
            anomalies.append("High concentration of off-hours activity")
        
        return anomalies
    
    def _detect_frequency_anomalies(self, frequency_patterns: Dict) -> List[str]:
        """
        Detect frequency anomalies
        """
        anomalies = []
        
        # Check for very short intervals (burst activity)
        min_interval = frequency_patterns.get('min_interval_hours', float('inf'))
        if min_interval < 0.1:  # Less than 6 minutes between reports
            anomalies.append("Burst activity detected (very short intervals)")
        
        # Check for high variance in intervals (irregular pattern)
        variance = frequency_patterns.get('interval_variance', 0)
        avg_interval = frequency_patterns.get('average_interval_hours', 1)
        
        if variance > avg_interval ** 2:  # High relative variance
            anomalies.append("Highly irregular reporting pattern")
        
        return anomalies
    
    def _detect_geographic_anomalies(self, geographic_patterns: Dict) -> List[str]:
        """
        Detect geographic anomalies
        """
        anomalies = []
        
        # Check for impossible speeds
        suspicious_speeds = geographic_patterns.get('suspicious_speeds', 0)
        if suspicious_speeds > 0:
            anomalies.append(f"Impossible travel speeds detected ({suspicious_speeds} instances)")
        
        # Check for excessive location changes
        location_changes = geographic_patterns.get('location_changes', 0)
        if location_changes > 50:  # Arbitrary threshold
            anomalies.append("Excessive location changes detected")
        
        # Check for too many countries
        unique_countries = geographic_patterns.get('unique_countries', 0)
        if unique_countries > 10:
            anomalies.append("Activity across many countries")
        
        return anomalies
    
    def _detect_pattern_breaks(self, patterns: Dict) -> List[str]:
        """
        Detect sudden breaks in established patterns
        """
        breaks = []
        
        # This would implement change point detection algorithms
        # For now, return basic pattern break detection
        
        if 'frequency' in patterns:
            # Check if recent activity is very different from historical
            recent_variance = patterns['frequency'].get('interval_variance', 0)
            if recent_variance > 100:  # High variance indicates pattern break
                breaks.append("Pattern break in activity frequency")
        
        return breaks
    
    def _calculate_behavioral_risk(self, anomalies: List[str], temporal_analysis: Dict, 
                                 communication_analysis: Dict) -> float:
        """
        Calculate overall behavioral risk score
        """
        risk_score = 0.0
        
        # Base risk from anomalies
        anomaly_risk = min(1.0, len(anomalies) * 0.2)
        
        # Risk from temporal patterns
        temporal_risk = temporal_analysis.get('risk_score', 0.0)
        
        # Risk from communication patterns
        communication_risk = communication_analysis.get('risk_score', 0.0)
        
        # Combined risk (weighted average)
        risk_score = (anomaly_risk * 0.4 + temporal_risk * 0.4 + communication_risk * 0.2)
        
        return min(1.0, risk_score)
    
    def _calculate_pattern_confidence(self, patterns: Dict) -> float:
        """
        Calculate confidence in pattern analysis
        """
        # Base confidence on amount of data available
        total_data_points = 0
        
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict):
                total_data_points += len(pattern_data)
            elif isinstance(pattern_data, list):
                total_data_points += len(pattern_data)
            else:
                total_data_points += 1
        
        # Confidence increases with more data, plateaus at high values
        confidence = min(1.0, total_data_points / 50.0)
        
        return confidence
    
    def _calculate_distance(self, coord1: tuple, coord2: tuple) -> float:
        """
        Calculate distance between coordinates (reuse from risk_calculator)
        """
        import math
        
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