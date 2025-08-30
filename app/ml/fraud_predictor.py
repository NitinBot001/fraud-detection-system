import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import joblib
import asyncio
from datetime import datetime, timedelta
from app.utils.logger import get_logger
from app.services.cache_service import CacheService

logger = get_logger(__name__)

class FraudPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.cache = CacheService()
        
        # Feature definitions
        self.numerical_features = [
            'call_frequency', 'call_duration_avg', 'network_size',
            'location_changes', 'report_count', 'account_age_days',
            'device_changes', 'identity_links', 'verification_score'
        ]
        
        self.categorical_features = [
            'carrier', 'country', 'device_type', 'registration_method'
        ]
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000
                }
            }
        }
    
    async def predict_fraud_probability(self, features: Dict) -> float:
        """
        Predict fraud probability using ensemble of models
        """
        try:
            # Prepare features
            feature_vector = await self._prepare_prediction_features(features)
            
            if feature_vector is None:
                return 0.0
            
            # Get predictions from all models
            predictions = []
            
            for model_name in self.model_configs.keys():
                if model_name in self.models:
                    try:
                        # Get probability of fraud class
                        prob = self.models[model_name].predict_proba(feature_vector)[0][1]
                        predictions.append(prob)
                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_name}: {str(e)}")
            
            if not predictions:
                return 0.0
            
            # Ensemble prediction (weighted average)
            weights = {'random_forest': 0.4, 'gradient_boosting': 0.4, 'logistic_regression': 0.2}
            
            if len(predictions) == len(self.model_configs):
                weighted_pred = sum(pred * weights[model] for pred, model in 
                                  zip(predictions, self.model_configs.keys()))
            else:
                weighted_pred = np.mean(predictions)
            
            return min(1.0, max(0.0, weighted_pred))
            
        except Exception as e:
            logger.error(f"Fraud prediction failed: {str(e)}")
            return 0.0
    
    async def _prepare_prediction_features(self, features: Dict) -> Optional[np.ndarray]:
        """
        Prepare features for prediction
        """
        try:
            # Extract numerical features
            numerical_values = []
            for feature in self.numerical_features:
                value = features.get(feature, 0.0)
                if isinstance(value, (list, dict)):
                    value = len(value) if isinstance(value, list) else sum(value.values())
                numerical_values.append(float(value))
            
            # Extract and encode categorical features
            categorical_values = []
            for feature in self.categorical_features:
                value = features.get(feature, 'unknown')
                
                encoder_key = f'{feature}_encoder'
                if encoder_key in self.encoders:
                    try:
                        encoded_value = self.encoders[encoder_key].transform([str(value)])[0]
                    except ValueError:
                        # Handle unseen categories
                        encoded_value = 0
                else:
                    encoded_value = 0
                
                categorical_values.append(encoded_value)
            
            # Combine features
            all_features = numerical_values + categorical_values
            feature_vector = np.array(all_features).reshape(1, -1)
            
            # Scale features
            scaler_key = 'prediction_scaler'
            if scaler_key in self.scalers:
                feature_vector = self.scalers[scaler_key].transform(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            return None
    
    async def train_models(self, training_data: pd.DataFrame, target_column: str = 'is_fraud') -> Dict:
        """
        Train fraud prediction models
        """
        try:
            logger.info("Training fraud prediction models")
            
            if target_column not in training_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in training data")
            
            # Prepare features and target
            X, y = await self._prepare_training_data(training_data, target_column)
            
            if X is None or len(X) == 0:
                raise ValueError("No valid training data available")
            
            # Train all models
            model_metrics = {}
            
            for model_name, config in self.model_configs.items():
                logger.info(f"Training {model_name}")
                
                # Initialize and train model
                model = config['model'](**config['params'])
                model.fit(X, y)
                
                # Store trained model
                self.models[model_name] = model
                
                # Calculate cross-validation metrics
                cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5), 
                                          scoring='roc_auc')
                
                model_metrics[model_name] = {
                    'cv_auc_mean': np.mean(cv_scores),
                    'cv_auc_std': np.std(cv_scores),
                    'feature_importance': self._get_feature_importance(model, model_name)
                }
                
                logger.info(f"{model_name} - CV AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            # Calculate ensemble metrics
            ensemble_metrics = await self._calculate_ensemble_metrics(X, y)
            model_metrics['ensemble'] = ensemble_metrics
            
            logger.info("Fraud prediction models trained successfully")
            return model_metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {}
    
    async def _prepare_training_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with feature engineering
        """
        try:
            # Create feature columns if they don't exist
            for feature in self.numerical_features:
                if feature not in data.columns:
                    data[feature] = 0.0
            
            for feature in self.categorical_features:
                if feature not in data.columns:
                    data[feature] = 'unknown'
            
            # Extract numerical features
            X_numerical = data[self.numerical_features].fillna(0).values
            
            # Encode categorical features
            X_categorical = []
            for feature in self.categorical_features:
                encoder_key = f'{feature}_encoder'
                
                if encoder_key not in self.encoders:
                    self.encoders[encoder_key] = LabelEncoder()
                
                # Fit encoder on training data
                encoded_values = self.encoders[encoder_key].fit_transform(
                    data[feature].fillna('unknown').astype(str)
                )
                X_categorical.append(encoded_values.reshape(-1, 1))
            
            # Combine features
            if X_categorical:
                X_categorical = np.hstack(X_categorical)
                X = np.hstack([X_numerical, X_categorical])
            else:
                X = X_numerical
            
            # Scale features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scalers['prediction_scaler'] = scaler
            
            # Extract target
            y = data[target_column].values
            
            return X, y
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {str(e)}")
            return None, None
    
    def _get_feature_importance(self, model, model_name: str) -> Dict:
        """
        Get feature importance from trained model
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.numerical_features + self.categorical_features
                
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # For linear models
                importances = np.abs(model.coef_[0])
                feature_names = self.numerical_features + self.categorical_features
                
                return dict(zip(feature_names, importances))
            else:
                return {}
                
        except Exception as e:
            logger.warning(f"Failed to get feature importance for {model_name}: {str(e)}")
            return {}
    
    async def _calculate_ensemble_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calculate metrics for ensemble prediction
        """
        try:
            # Get predictions from all models
            predictions = []
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]
                    predictions.append(pred_proba)
            
            if not predictions:
                return {}
            
            # Ensemble prediction
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Calculate AUC
            auc_score = roc_auc_score(y, ensemble_pred)
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y, ensemble_pred)
            pr_auc = np.trapz(recall, precision)
            
            return {
                'auc_score': auc_score,
                'pr_auc': pr_auc,
                'n_models': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Ensemble metrics calculation failed: {str(e)}")
            return {}
    
    async def predict_batch(self, feature_list: List[Dict]) -> List[float]:
        """
        Predict fraud probability for a batch of phone numbers
        """
        try:
            predictions = []
            
            # Process in batches for efficiency
            batch_size = 100
            for i in range(0, len(feature_list), batch_size):
                batch = feature_list[i:i + batch_size]
                
                batch_predictions = await asyncio.gather(*[
                    self.predict_fraud_probability(features) for features in batch
                ])
                
                predictions.extend(batch_predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return [0.0] * len(feature_list)
    
    async def get_prediction_explanation(self, features: Dict) -> Dict:
        """
        Get explanation for fraud prediction
        """
        try:
            # Get base prediction
            fraud_prob = await self.predict_fraud_probability(features)
            
            # Feature contribution analysis
            feature_vector = await self._prepare_prediction_features(features)
            
            if feature_vector is None:
                return {'fraud_probability': fraud_prob, 'explanations': []}
            
            explanations = []
            
            # Analyze feature contributions
            feature_names = self.numerical_features + self.categorical_features
            
            for i, (feature_name, value) in enumerate(zip(feature_names, feature_vector[0])):
                if feature_name in features:
                    original_value = features[feature_name]
                    
                    # Determine if this feature increases fraud risk
                    risk_contribution = self._analyze_feature_risk(feature_name, original_value)
                    
                    if risk_contribution > 0.1:  # Significant contribution
                        explanations.append({
                            'feature': feature_name,
                            'value': original_value,
                            'risk_contribution': risk_contribution,
                            'explanation': self._get_feature_explanation(feature_name, original_value)
                        })
            
            # Sort by risk contribution
            explanations.sort(key=lambda x: x['risk_contribution'], reverse=True)
            
            return {
                'fraud_probability': fraud_prob,
                'explanations': explanations[:5],  # Top 5 contributing factors
                'model_confidence': self._calculate_prediction_confidence(feature_vector)
            }
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {str(e)}")
            return {'fraud_probability': 0.0, 'explanations': []}
    
    def _analyze_feature_risk(self, feature_name: str, value) -> float:
        """
        Analyze how much a feature contributes to fraud risk
        """
        # This would ideally use SHAP values or similar explainability methods
        # For now, use simple heuristics
        
        risk_thresholds = {
            'report_count': {'high_risk': 5, 'weight': 0.8},
            'location_changes': {'high_risk': 20, 'weight': 0.6},
            'device_changes': {'high_risk': 5, 'weight': 0.7},
            'network_size': {'high_risk': 100, 'weight': 0.5},
            'verification_score': {'low_risk': 0.8, 'weight': 0.6}  # Lower is worse
        }
        
        if feature_name in risk_thresholds:
            threshold_info = risk_thresholds[feature_name]
            
            if 'high_risk' in threshold_info:
                if isinstance(value, (int, float)) and value >= threshold_info['high_risk']:
                    return threshold_info['weight']
            elif 'low_risk' in threshold_info:
                if isinstance(value, (int, float)) and value <= threshold_info['low_risk']:
                    return threshold_info['weight']
        
        return 0.0
    
    def _get_feature_explanation(self, feature_name: str, value) -> str:
        """
        Get human-readable explanation for feature contribution
        """
        explanations = {
            'report_count': f"High number of fraud reports ({value})",
            'location_changes': f"Frequent location changes ({value})",
            'device_changes': f"Multiple device changes ({value})",
            'network_size': f"Large network size ({value})",
            'verification_score': f"Low identity verification score ({value})",
            'call_frequency': f"Unusual call frequency pattern ({value})"
        }
        
        return explanations.get(feature_name, f"Anomalous {feature_name}: {value}")
    
    def _calculate_prediction_confidence(self, feature_vector: np.ndarray) -> float:
        """
        Calculate confidence in the prediction
        """
        try:
            # Get prediction variance across models
            predictions = []
            
            for model in self.models.values():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(feature_vector)[0][1]
                    predictions.append(pred)
            
            if len(predictions) < 2:
                return 0.5
            
            # Lower variance = higher confidence
            variance = np.var(predictions)
            confidence = 1.0 / (1.0 + variance * 10)  # Scale variance to confidence
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    async def save_models(self, model_path: str):
        """
        Save trained models to disk
        """
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'numerical_features': self.numerical_features,
                'categorical_features': self.categorical_features,
                'timestamp': datetime.now()
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Fraud prediction models saved to {model_path}")
            
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
            self.encoders = model_data.get('encoders', {})
            self.numerical_features = model_data.get('numerical_features', self.numerical_features)
            self.categorical_features = model_data.get('categorical_features', self.categorical_features)
            
            logger.info(f"Fraud prediction models loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")