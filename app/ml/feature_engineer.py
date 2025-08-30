import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import (PhoneNumber, FraudReport, NetworkConnection, 
                               GeolocationData, BehaviorPattern, IdentityLink)
from app.utils.logger import get_logger
import networkx as nx
from collections import Counter, defaultdict
import math

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, db_session: Session):
        self.db = db_session
        
        # Feature categories
        self.temporal_features = [
            'account_age_days', 'days_since_last_report', 'report_frequency_30d',
            'report_frequency_7d', 'activity_hours_variance', 'weekend_activity_ratio'
        ]
        
        self.network_features = [
            'network_size', 'network_density', 'clustering_coefficient',
            'betweenness_centrality', 'degree_centrality', 'connected_components'
        ]
        
        self.geographic_features = [
            'unique_locations', 'travel_distance_total', 'impossible_speeds',
            'location_entropy', 'country_changes', 'location_frequency_variance'
        ]
        
        self.behavioral_features = [
            'report_severity_trend', 'identity_verification_score',
            'device_change_frequency', 'communication_pattern_score'
        ]
    
    async def engineer_comprehensive_features(self, phone_record: PhoneNumber, 
                                            window_days: int = 90) -> Dict[str, Any]:
        """
        Engineer comprehensive feature set for a phone number
        """
        try:
            features = {}
            cutoff_date = datetime.now() - timedelta(days=window_days)
            
            # Basic phone information features
            features.update(await self._extract_basic_features(phone_record))
            
            # Temporal features
            features.update(await self._extract_temporal_features(phone_record, cutoff_date))
            
            # Network features
            features.update(await self._extract_network_features(phone_record))
            
            # Geographic features
            features.update(await self._extract_geographic_features(phone_record, cutoff_date))
            
            # Behavioral features
            features.update(await self._extract_behavioral_features(phone_record, cutoff_date))
            
            # Report-based features
            features.update(await self._extract_report_features(phone_record, cutoff_date))
            
            # Identity features
            features.update(await self._extract_identity_features(phone_record))
            
            # Advanced derived features
            features.update(await self._extract_derived_features(features, phone_record))
            
            # Feature validation and cleaning
            features = self._validate_and_clean_features(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed for {phone_record.number}: {str(e)}")
            return {}
    
    async def _extract_basic_features(self, phone_record: PhoneNumber) -> Dict[str, Any]:
        """
        Extract basic phone information features
        """
        features = {}
        
        # Account age
        if phone_record.created_at:
            account_age = (datetime.now() - phone_record.created_at).days
            features['account_age_days'] = account_age
        else:
            features['account_age_days'] = 0
        
        # Phone type and carrier information
        features['is_mobile'] = 1 if phone_record.is_mobile else 0
        features['has_carrier_info'] = 1 if phone_record.carrier else 0
        features['has_location_info'] = 1 if phone_record.location else 0
        features['is_active'] = 1 if phone_record.is_active else 0
        
        # Country code analysis
        if phone_record.country_code:
            features['country_code_numeric'] = int(phone_record.country_code) if phone_record.country_code.isdigit() else 0
            features['is_common_country'] = 1 if phone_record.country_code in ['1', '44', '91', '86'] else 0
        else:
            features['country_code_numeric'] = 0
            features['is_common_country'] = 0
        
        return features
    
    async def _extract_temporal_features(self, phone_record: PhoneNumber, 
                                       cutoff_date: datetime) -> Dict[str, Any]:
        """
        Extract temporal pattern features
        """
        features = {}
        
        # Get reports within time window
        reports = self.db.query(FraudReport).filter(
            FraudReport.phone_number_id == phone_record.id,
            FraudReport.created_at >= cutoff_date
        ).order_by(FraudReport.created_at).all()
        
        if not reports:
            # Default values for no reports
            features.update({
                'total_reports': 0,
                'days_since_last_report': 999,
                'report_frequency_30d': 0,
                'report_frequency_7d': 0,
                'activity_hours_variance': 0,
                'weekend_activity_ratio': 0,
                'report_time_entropy': 0
            })
            return features
        
        # Basic temporal metrics
        features['total_reports'] = len(reports)
        
        last_report_date = max(r.created_at for r in reports)
        features['days_since_last_report'] = (datetime.now() - last_report_date).days
        
        # Report frequency in different windows
        now = datetime.now()
        reports_30d = [r for r in reports if r.created_at > now - timedelta(days=30)]
        reports_7d = [r for r in reports if r.created_at > now - timedelta(days=7)]
        
        features['report_frequency_30d'] = len(reports_30d)
        features['report_frequency_7d'] = len(reports_7d)
        
        # Time-of-day analysis
        report_hours = [r.created_at.hour for r in reports]
        if report_hours:
            features['activity_hours_variance'] = np.var(report_hours)
            features['activity_hours_mean'] = np.mean(report_hours)
            
            # Weekend activity ratio
            weekend_reports = sum(1 for r in reports if r.created_at.weekday() >= 5)
            features['weekend_activity_ratio'] = weekend_reports / len(reports)
            
            # Temporal entropy
            hour_counts = Counter(report_hours)
            total_reports = len(reports)
            entropy = -sum((count/total_reports) * math.log2(count/total_reports) 
                          for count in hour_counts.values() if count > 0)
            features['report_time_entropy'] = entropy
        else:
            features.update({
                'activity_hours_variance': 0,
                'activity_hours_mean': 12,
                'weekend_activity_ratio': 0,
                'report_time_entropy': 0
            })
        
        # Report interval analysis
        if len(reports) > 1:
            intervals = []
            sorted_reports = sorted(reports, key=lambda x: x.created_at)
            
            for i in range(1, len(sorted_reports)):
                interval_hours = (sorted_reports[i].created_at - sorted_reports[i-1].created_at).total_seconds() / 3600
                intervals.append(interval_hours)
            
            features['avg_report_interval_hours'] = np.mean(intervals)
            features['min_report_interval_hours'] = min(intervals)
            features['report_interval_variance'] = np.var(intervals)
            features['burst_reports'] = sum(1 for interval in intervals if interval < 1)  # Reports within 1 hour
        else:
            features.update({
                'avg_report_interval_hours': 0,
                'min_report_interval_hours': 0,
                'report_interval_variance': 0,
                'burst_reports': 0
            })
        
        return features
    
    async def _extract_network_features(self, phone_record: PhoneNumber) -> Dict[str, Any]:
        """
        Extract network topology features
        """
        features = {}
        
        # Get direct connections
        connections = self.db.query(NetworkConnection).filter(
            (NetworkConnection.source_phone_id == phone_record.id) |
            (NetworkConnection.target_phone_id == phone_record.id)
        ).all()
        
        features['direct_connections'] = len(connections)
        
        if not connections:
            # Default values for isolated nodes
            features.update({
                'network_size': 0,
                'network_density': 0,
                'clustering_coefficient': 0,
                'betweenness_centrality': 0,
                'degree_centrality': 0,
                'avg_connection_strength': 0,
                'max_connection_strength': 0,
                'connection_types_count': 0
            })
            return features
        
        # Build local network graph
        graph = nx.Graph()
        graph.add_node(phone_record.number)
        
        connected_phones = set()
        connection_types = set()
        strengths = []
        
        for conn in connections:
            # Get connected phone
            if conn.source_phone_id == phone_record.id:
                other_phone = self.db.query(PhoneNumber).filter_by(id=conn.target_phone_id).first()
            else:
                other_phone = self.db.query(PhoneNumber).filter_by(id=conn.source_phone_id).first()
            
            if other_phone:
                connected_phones.add(other_phone.number)
                connection_types.add(conn.connection_type)
                strengths.append(conn.strength)
                
                graph.add_edge(phone_record.number, other_phone.number, 
                             weight=conn.strength, type=conn.connection_type)
        
        # Calculate network metrics
        features['network_size'] = len(connected_phones)
        features['connection_types_count'] = len(connection_types)
        
        if strengths:
            features['avg_connection_strength'] = np.mean(strengths)
            features['max_connection_strength'] = max(strengths)
            features['connection_strength_variance'] = np.var(strengths)
        else:
            features.update({
                'avg_connection_strength': 0,
                'max_connection_strength': 0,
                'connection_strength_variance': 0
            })
        
        # Graph-based metrics
        if len(graph.nodes()) > 1:
            try:
                # Density
                features['network_density'] = nx.density(graph)
                
                # Centrality measures
                degree_centrality = nx.degree_centrality(graph)
                features['degree_centrality'] = degree_centrality.get(phone_record.number, 0)
                
                if len(graph.nodes()) > 2:
                    betweenness_centrality = nx.betweenness_centrality(graph)
                    features['betweenness_centrality'] = betweenness_centrality.get(phone_record.number, 0)
                    
                    clustering_coefficient = nx.clustering(graph)
                    features['clustering_coefficient'] = clustering_coefficient.get(phone_record.number, 0)
                else:
                    features['betweenness_centrality'] = 0
                    features['clustering_coefficient'] = 0
                
            except Exception as e:
                logger.warning(f"Network metrics calculation failed: {str(e)}")
                features.update({
                    'network_density': 0,
                    'degree_centrality': 0,
                    'betweenness_centrality': 0,
                    'clustering_coefficient': 0
                })
        else:
            features.update({
                'network_density': 0,
                'degree_centrality': 0,
                'betweenness_centrality': 0,
                'clustering_coefficient': 0
            })
        
        return features
    
    async def _extract_geographic_features(self, phone_record: PhoneNumber, 
                                         cutoff_date: datetime) -> Dict[str, Any]:
        """
        Extract geographic mobility features
        """
        features = {}
        
        # Get location data within time window
        locations = self.db.query(GeolocationData).filter(
            GeolocationData.phone_number_id == phone_record.id,
            GeolocationData.timestamp >= cutoff_date
        ).order_by(GeolocationData.timestamp).all()
        
        if not locations:
            # Default values for no location data
            features.update({
                'unique_locations': 0,
                'unique_countries': 0,
                'unique_cities': 0,
                'travel_distance_total': 0,
                'impossible_speeds': 0,
                'location_entropy': 0,
                'avg_accuracy': 0,
                'location_changes': 0
            })
            return features
        
        # Basic location metrics
        unique_countries = set(loc.country for loc in locations if loc.country)
        unique_cities = set(f"{loc.country}_{loc.city}" for loc in locations if loc.country and loc.city)
        
        features['unique_locations'] = len(locations)
        features['unique_countries'] = len(unique_countries)
        features['unique_cities'] = len(unique_cities)
        
        # Accuracy metrics
        accuracies = [loc.accuracy for loc in locations if loc.accuracy]
        features['avg_accuracy'] = np.mean(accuracies) if accuracies else 0
        
        # Movement analysis
        movements = []
        impossible_speeds = 0
        total_distance = 0
        
        for i in range(1, len(locations)):
            loc1, loc2 = locations[i-1], locations[i]
            
            if (loc1.latitude and loc1.longitude and 
                loc2.latitude and loc2.longitude):
                
                # Calculate distance
                distance = self._calculate_distance(
                    (loc1.latitude, loc1.longitude),
                    (loc2.latitude, loc2.longitude)
                )
                
                # Calculate time difference
                time_diff = (loc2.timestamp - loc1.timestamp).total_seconds() / 3600  # hours
                
                if time_diff > 0:
                    speed = distance / time_diff
                    movements.append({
                        'distance': distance,
                        'time': time_diff,
                        'speed': speed
                    })
                    
                    total_distance += distance
                    
                    # Check for impossible speeds (>1000 km/h)
                    if speed > 1000:
                        impossible_speeds += 1
        
        features['travel_distance_total'] = total_distance
        features['impossible_speeds'] = impossible_speeds
        features['location_changes'] = len(movements)
        
        if movements:
            speeds = [m['speed'] for m in movements]
            features['avg_travel_speed'] = np.mean(speeds)
            features['max_travel_speed'] = max(speeds)
            features['travel_speed_variance'] = np.var(speeds)
        else:
            features.update({
                'avg_travel_speed': 0,
                'max_travel_speed': 0,
                'travel_speed_variance': 0
            })
        
        # Location entropy (diversity of locations)
        if unique_countries:
            country_counts = Counter(loc.country for loc in locations if loc.country)
            total_locations = len(locations)
            entropy = -sum((count/total_locations) * math.log2(count/total_locations) 
                          for count in country_counts.values() if count > 0)
            features['location_entropy'] = entropy
        else:
            features['location_entropy'] = 0
        
        return features
    
    async def _extract_behavioral_features(self, phone_record: PhoneNumber, 
                                         cutoff_date: datetime) -> Dict[str, Any]:
        """
        Extract behavioral pattern features
        """
        features = {}
        
        # Get behavior patterns
        patterns = self.db.query(BehaviorPattern).filter(
            BehaviorPattern.phone_number_id == phone_record.id,
            BehaviorPattern.created_at >= cutoff_date
        ).all()
        
        features['behavior_patterns_count'] = len(patterns)
        
        if patterns:
            # Analyze pattern types
            pattern_types = Counter(p.pattern_type for p in patterns)
            features['unique_pattern_types'] = len(pattern_types)
            
            # Confidence scores
            confidences = [p.confidence for p in patterns if p.confidence is not None]
            if confidences:
                features['avg_pattern_confidence'] = np.mean(confidences)
                features['min_pattern_confidence'] = min(confidences)
            else:
                features.update({
                    'avg_pattern_confidence': 0,
                    'min_pattern_confidence': 0
                })
            
            # Anomaly detection
            anomalies = sum(1 for p in patterns if p.anomaly_detected)
            features['behavioral_anomalies'] = anomalies
            features['anomaly_ratio'] = anomalies / len(patterns)
        else:
            features.update({
                'unique_pattern_types': 0,
                'avg_pattern_confidence': 0,
                'min_pattern_confidence': 0,
                'behavioral_anomalies': 0,
                'anomaly_ratio': 0
            })
        
        return features
    
    async def _extract_report_features(self, phone_record: PhoneNumber, 
                                     cutoff_date: datetime) -> Dict[str, Any]:
        """
        Extract fraud report-based features
        """
        features = {}
        
        # Get reports within time window
        reports = self.db.query(FraudReport).filter(
            FraudReport.phone_number_id == phone_record.id,
            FraudReport.created_at >= cutoff_date
        ).all()
        
        if not reports:
            features.update({
                'fraud_types_count': 0,
                'verified_reports_ratio': 0,
                'avg_severity_score': 0,
                'severity_trend': 0,
                'confidence_score_avg': 0
            })
            return features
        
        # Fraud type diversity
        fraud_types = set(r.fraud_type for r in reports if r.fraud_type)
        features['fraud_types_count'] = len(fraud_types)
        
        # Verification status analysis
        verified_reports = [r for r in reports if r.status == 'VERIFIED']
        features['verified_reports_ratio'] = len(verified_reports) / len(reports)
        
        # Severity analysis
        severity_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        severity_scores = [severity_map.get(r.severity, 2) for r in reports]
        
        if severity_scores:
            features['avg_severity_score'] = np.mean(severity_scores)
            
            # Severity trend (increasing, decreasing, stable)
            if len(severity_scores) > 1:
                trend = np.polyfit(range(len(severity_scores)), severity_scores, 1)[0]
                features['severity_trend'] = trend
            else:
                features['severity_trend'] = 0
        else:
            features.update({
                'avg_severity_score': 0,
                'severity_trend': 0
            })
        
        # Confidence scores
        confidence_scores = [r.confidence_score for r in reports if r.confidence_score is not None]
        features['confidence_score_avg'] = np.mean(confidence_scores) if confidence_scores else 0
        
        return features
    
    async def _extract_identity_features(self, phone_record: PhoneNumber) -> Dict[str, Any]:
        """
        Extract identity verification features
        """
        features = {}
        
        # Get identity links
        identity_links = self.db.query(IdentityLink).filter_by(
            phone_number_id=phone_record.id
        ).all()
        
        features['identity_links_count'] = len(identity_links)
        
        if not identity_links:
            features.update({
                'identity_types_count': 0,
                'verified_identities_ratio': 0,
                'avg_identity_confidence': 0,
                'has_government_id': 0,
                'has_email_verification': 0
            })
            return features
        
        # Identity type analysis
        identity_types = set(link.identity_type for link in identity_links)
        features['identity_types_count'] = len(identity_types)
        
        # Verification status
        verified_identities = [link for link in identity_links if link.verification_status == 'VERIFIED']
        features['verified_identities_ratio'] = len(verified_identities) / len(identity_links)
        
        # Confidence scores
        confidence_scores = [link.confidence_score for link in identity_links if link.confidence_score is not None]
        features['avg_identity_confidence'] = np.mean(confidence_scores) if confidence_scores else 0
        
        # Specific identity types
        features['has_government_id'] = 1 if any(link.identity_type in ['AADHAR', 'SSN', 'PASSPORT'] for link in identity_links) else 0
        features['has_email_verification'] = 1 if any(link.identity_type == 'EMAIL' for link in identity_links) else 0
        features['has_device_id'] = 1 if any(link.identity_type == 'DEVICE_ID' for link in identity_links) else 0
        
        return features
    
    async def _extract_derived_features(self, features: Dict[str, Any], 
                                      phone_record: PhoneNumber) -> Dict[str, Any]:
        """
        Extract derived and composite features
        """
        derived = {}
        
        # Risk ratios and combinations
        if features.get('total_reports', 0) > 0 and features.get('account_age_days', 0) > 0:
            derived['reports_per_day'] = features['total_reports'] / features['account_age_days']
        else:
            derived['reports_per_day'] = 0
        
        # Network to reports ratio
        if features.get('network_size', 0) > 0:
            derived['reports_per_connection'] = features.get('total_reports', 0) / features['network_size']
        else:
            derived['reports_per_connection'] = 0
        
        # Movement to time ratio
        if features.get('account_age_days', 0) > 0:
            derived['locations_per_day'] = features.get('unique_locations', 0) / features['account_age_days']
        else:
            derived['locations_per_day'] = 0
        
        # Composite risk scores
        mobility_risk = min(1.0, (
            features.get('impossible_speeds', 0) * 0.4 +
            features.get('unique_countries', 0) / 10.0 * 0.3 +
            features.get('travel_distance_total', 0) / 10000.0 * 0.3
        ))
        derived['mobility_risk_score'] = mobility_risk
        
        network_risk = min(1.0, (
            features.get('network_size', 0) / 100.0 * 0.5 +
            features.get('clustering_coefficient', 0) * 0.3 +
            features.get('betweenness_centrality', 0) * 0.2
        ))
        derived['network_risk_score'] = network_risk
        
        temporal_risk = min(1.0, (
            features.get('report_frequency_7d', 0) / 5.0 * 0.4 +
            features.get('burst_reports', 0) / 5.0 * 0.3 +
            (1 - features.get('avg_report_interval_hours', 24) / 24.0) * 0.3
        ))
        derived['temporal_risk_score'] = temporal_risk
        
        return derived
    
    def _validate_and_clean_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean feature values
        """
        cleaned = {}
        
        for key, value in features.items():
            # Handle missing or invalid values
            if value is None:
                cleaned[key] = 0.0
            elif isinstance(value, (int, float)):
                # Handle infinite or NaN values
                if math.isnan(value) or math.isinf(value):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = float(value)
            elif isinstance(value, bool):
                cleaned[key] = 1.0 if value else 0.0
            else:
                # Convert other types to float if possible
                try:
                    cleaned[key] = float(value)
                except (ValueError, TypeError):
                    cleaned[key] = 0.0
        
        return cleaned
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
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
    
    async def engineer_batch_features(self, phone_numbers: List[str], 
                                    window_days: int = 90) -> pd.DataFrame:
        """
        Engineer features for a batch of phone numbers
        """
        try:
            feature_list = []
            
            for phone_number in phone_numbers:
                # Get phone record
                phone_record = self.db.query(PhoneNumber).filter_by(number=phone_number).first()
                
                if phone_record:
                    features = await self.engineer_comprehensive_features(phone_record, window_days)
                    features['phone_number'] = phone_number
                    feature_list.append(features)
                else:
                    # Create default feature set for unknown numbers
                    default_features = self._get_default_features()
                    default_features['phone_number'] = phone_number
                    feature_list.append(default_features)
            
            # Convert to DataFrame
            df = pd.DataFrame(feature_list)
            
            # Fill any remaining NaN values
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Batch feature engineering failed: {str(e)}")
            return pd.DataFrame()
    
    def _get_default_features(self) -> Dict[str, Any]:
        """
        Get default feature values for unknown phone numbers
        """
        default_features = {}
        
        # Set all possible features to default values
        feature_names = (self.temporal_features + self.network_features + 
                        self.geographic_features + self.behavioral_features)
        
        for feature in feature_names:
            default_features[feature] = 0.0
        
        # Add other common features
        additional_features = [
            'account_age_days', 'is_mobile', 'has_carrier_info', 'total_reports',
            'fraud_types_count', 'identity_links_count', 'verified_identities_ratio'
        ]
        
        for feature in additional_features:
            default_features[feature] = 0.0
        
        return default_features