import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import asyncio
from datetime import datetime, timedelta
from app.utils.logger import get_logger
from app.services.cache_service import CacheService

logger = get_logger(__name__)

class AnomalyDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.cache = CacheService()
        self.feature_columns = [
            'call_frequency', 'call_duration_avg', 'call_time_variance',
            'location_changes', 'speed_violations', 'network_size',
            'report_count', 'severity_score', 'temporal_entropy',
            'geographic_entropy', 'identity_score', 'device_changes'
        ]
        
        # Model parameters
        self.isolation_forest_params = {
            'contamination': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        self.dbscan_params = {
            'eps': 0.5,
            'min_samples': 5
        }
    
    async def detect_anomalies(self, features: Dict) -> float:
        """
        Detect anomalies using multiple ML algorithms
        """
        try:
            # Convert features to numpy array
            feature_vector = await self._prepare_features(features)
            
            if feature_vector is None:
                return 0.0
            
            # Run multiple anomaly detection algorithms
            isolation_score = await self._isolation_forest_detection(feature_vector)
            clustering_score = await self._clustering_based_detection(feature_vector)
            statistical_score = await self._statistical_anomaly_detection(feature_vector)
            
            # Ensemble scoring
            ensemble_score = await self._ensemble_anomaly_score(
                isolation_score, clustering_score, statistical_score
            )
            
            return min(1.0, max(0.0, ensemble_score))
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return 0.0
    
    async def _prepare_features(self, features: Dict) -> Optional[np.ndarray]:
        """
        Prepare and normalize features for anomaly detection
        """
        try:
            # Extract feature values with defaults
            feature_values = []
            for col in self.feature_columns:
                value = features.get(col, 0.0)
                # Handle nested dictionaries
                if isinstance(value, dict):
                    value = sum(value.values()) if value else 0.0
                elif isinstance(value, list):
                    value = len(value)
                feature_values.append(float(value))
            
            feature_vector = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            scaler_key = 'anomaly_scaler'
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                # Initialize with some default scaling
                dummy_data = np.random.normal(0, 1, (100, len(self.feature_columns)))
                self.scalers[scaler_key].fit(dummy_data)
            
            scaled_features = self.scalers[scaler_key].transform(feature_vector)
            return scaled_features
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            return None
    
    async def _isolation_forest_detection(self, feature_vector: np.ndarray) -> float:
        """
        Anomaly detection using Isolation Forest
        """
        try:
            model_key = 'isolation_forest'
            
            if model_key not in self.models:
                self.models[model_key] = IsolationForest(**self.isolation_forest_params)
                # Initialize with dummy data if no training data available
                dummy_data = np.random.normal(0, 1, (1000, feature_vector.shape[1]))
                self.models[model_key].fit(dummy_data)
            
            # Get anomaly score
            anomaly_score = self.models[model_key].decision_function(feature_vector)[0]
            
            # Convert to 0-1 scale (lower scores are more anomalous)
            normalized_score = max(0, min(1, (anomaly_score + 0.5) / 1.0))
            return 1 - normalized_score  # Invert so higher = more anomalous
            
        except Exception as e:
            logger.error(f"Isolation forest detection failed: {str(e)}")
            return 0.0
    
    async def _clustering_based_detection(self, feature_vector: np.ndarray) -> float:
        """
        Anomaly detection based on clustering
        """
        try:
            # Use cached cluster centers if available
            cache_key = "cluster_centers"
            cluster_centers = await self.cache.get(cache_key)
            
            if cluster_centers is None:
                # Generate some representative cluster centers
                cluster_centers = await self._generate_cluster_centers(feature_vector.shape[1])
                await self.cache.set(cache_key, cluster_centers, timeout=3600)
            
            # Calculate distance to nearest cluster center
            distances = [np.linalg.norm(feature_vector - center) for center in cluster_centers]
            min_distance = min(distances)
            
            # Normalize distance to anomaly score
            threshold = 2.0  # Distance threshold for anomaly
            anomaly_score = min(1.0, min_distance / threshold)
            
            return anomaly_score
            
        except Exception as e:
            logger.error(f"Clustering-based detection failed: {str(e)}")
            return 0.0
    
    async def _statistical_anomaly_detection(self, feature_vector: np.ndarray) -> float:
        """
        Statistical anomaly detection using z-scores
        """
        try:
            # Get cached statistics
            cache_key = "feature_statistics"
            stats = await self.cache.get(cache_key)
            
            if stats is None:
                # Use default statistics (would be learned from training data)
                stats = {
                    'means': np.zeros(feature_vector.shape[1]),
                    'stds': np.ones(feature_vector.shape[1])
                }
                await self.cache.set(cache_key, stats, timeout=3600)
            
            # Calculate z-scores
            z_scores = np.abs((feature_vector[0] - stats['means']) / (stats['stds'] + 1e-6))
            
            # Max z-score as anomaly indicator
            max_z_score = np.max(z_scores)
            
            # Convert to 0-1 scale
            anomaly_score = min(1.0, max_z_score / 3.0)  # 3-sigma rule
            
            return anomaly_score
            
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {str(e)}")
            return 0.0
    
    async def _ensemble_anomaly_score(self, isolation_score: float, 
                                    clustering_score: float, 
                                    statistical_score: float) -> float:
        """
        Combine multiple anomaly scores using ensemble method
        """
        # Weighted ensemble
        weights = {
            'isolation': 0.4,
            'clustering': 0.3,
            'statistical': 0.3
        }
        
        ensemble_score = (
            isolation_score * weights['isolation'] +
            clustering_score * weights['clustering'] +
            statistical_score * weights['statistical']
        )
        
        return ensemble_score
    
    async def _generate_cluster_centers(self, n_features: int) -> List[np.ndarray]:
        """
        Generate representative cluster centers for normal behavior
        """
        # This would ideally be learned from training data
        # For now, generate some reasonable cluster centers
        centers = []
        
        # Normal behavior cluster (low values)
        centers.append(np.random.normal(0, 0.5, n_features))
        
        # Moderate activity cluster
        centers.append(np.random.normal(0.5, 0.3, n_features))
        
        # High activity but legitimate cluster
        centers.append(np.random.normal(1.0, 0.4, n_features))
        
        return centers
    
    async def train_models(self, training_data: pd.DataFrame) -> Dict:
        """
        Train anomaly detection models with new data
        """
        try:
            logger.info("Training anomaly detection models")
            
            # Prepare training features
            X = training_data[self.feature_columns].values
            
            # Fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['anomaly_scaler'] = scaler
            
            # Train Isolation Forest
            iso_forest = IsolationForest(**self.isolation_forest_params)
            iso_forest.fit(X_scaled)
            self.models['isolation_forest'] = iso_forest
            
            # Generate cluster centers using DBSCAN
            dbscan = DBSCAN(**self.dbscan_params)
            cluster_labels = dbscan.fit_predict(X_scaled)
            
            # Calculate cluster centers
            unique_labels = set(cluster_labels)
            cluster_centers = []
            for label in unique_labels:
                if label != -1:  # Ignore noise points
                    cluster_mask = cluster_labels == label
                    center = np.mean(X_scaled[cluster_mask], axis=0)
                    cluster_centers.append(center)
            
            # Cache cluster centers
            await self.cache.set("cluster_centers", cluster_centers, timeout=86400)
            
            # Calculate and cache feature statistics
            stats = {
                'means': np.mean(X_scaled, axis=0),
                'stds': np.std(X_scaled, axis=0)
            }
            await self.cache.set("feature_statistics", stats, timeout=86400)
            
            # Calculate training metrics
            metrics = await self._calculate_training_metrics(X_scaled, cluster_labels)
            
            logger.info("Anomaly detection models trained successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {}
    
    async def _calculate_training_metrics(self, X: np.ndarray, cluster_labels: np.ndarray) -> Dict:
        """
        Calculate training metrics
        """
        try:
            metrics = {}
            
            # Silhouette score for clustering
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(X, cluster_labels)
                metrics['silhouette_score'] = silhouette
            
            # Number of clusters
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            metrics['n_clusters'] = n_clusters
            
            # Noise ratio
            noise_ratio = list(cluster_labels).count(-1) / len(cluster_labels)
            metrics['noise_ratio'] = noise_ratio
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            return {}
    
    async def save_models(self, model_path: str):
        """
        Save trained models to disk
        """
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'timestamp': datetime.now()
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Models saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
    
    async def load_models(self, model_path: str):
        """
        Load trained models from disk
        """
        try:
            model_data = joblib.load(model_path)
            
            self.models = model_data.get('models', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_columns = model_data.get('feature_columns', self.feature_columns)
            
            logger.info(f"Models loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")