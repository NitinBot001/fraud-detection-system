import re
import json
import hashlib
import phonenumbers
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)

def format_phone_number(phone: str, region: str = None) -> str:
    """
    Format phone number to standard international format
    """
    try:
        parsed = phonenumbers.parse(phone, region)
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        # Return cleaned number if parsing fails
        return re.sub(r'[^\d+]', '', phone)

def parse_phone_number(phone: str) -> Dict[str, Any]:
    """
    Parse phone number and extract metadata
    """
    try:
        parsed = phonenumbers.parse(phone, None)
        
        return {
            'number': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
            'country_code': parsed.country_code,
            'national_number': parsed.national_number,
            'country': phonenumbers.region_code_for_number(parsed),
            'carrier': phonenumbers.carrier.name_for_number(parsed, 'en'),
            'timezone': phonenumbers.timezone.time_zones_for_number(parsed),
            'number_type': phonenumbers.number_type(parsed).name,
            'is_valid': phonenumbers.is_valid_number(parsed),
            'is_possible': phonenumbers.is_possible_number(parsed)
        }
    except phonenumbers.NumberParseException as e:
        logger.warning(f"Phone number parsing failed: {str(e)}")
        return {
            'number': phone,
            'country_code': None,
            'is_valid': False,
            'error': str(e)
        }

def calculate_time_difference(dt1: datetime, dt2: datetime) -> Dict[str, Any]:
    """
    Calculate time difference with multiple units
    """
    diff = abs(dt2 - dt1)
    
    return {
        'total_seconds': diff.total_seconds(),
        'total_minutes': diff.total_seconds() / 60,
        'total_hours': diff.total_seconds() / 3600,
        'total_days': diff.days,
        'weeks': diff.days // 7,
        'human_readable': format_timedelta(diff)
    }

def format_timedelta(td: timedelta) -> str:
    """
    Format timedelta in human readable format
    """
    total_seconds = int(td.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes} minutes"
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        return f"{days}d {hours}h"

def generate_feature_hash(features: Dict[str, Any]) -> str:
    """
    Generate hash for feature vector (for caching ML predictions)
    """
    # Sort features by key for consistent hashing
    sorted_features = dict(sorted(features.items()))
    
    # Convert to JSON string
    feature_string = json.dumps(sorted_features, sort_keys=True, default=str)
    
    # Generate hash
    return hashlib.md5(feature_string.encode()).hexdigest()

def normalize_risk_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize risk score to specified range
    """
    return max(min_val, min(max_val, score))

def calculate_percentile_rank(value: float, data: List[float]) -> float:
    """
    Calculate percentile rank of value in dataset
    """
    if not data:
        return 0.0
    
    sorted_data = sorted(data)
    rank = sum(1 for x in sorted_data if x <= value)
    return rank / len(sorted_data)

def extract_domain_from_email(email: str) -> Optional[str]:
    """
    Extract domain from email address
    """
    try:
        return email.split('@')[1].lower()
    except (IndexError, AttributeError):
        return None

def classify_risk_level(score: float) -> str:
    """
    Classify numeric risk score into risk level
    """
    if score >= 0.9:
        return "CRITICAL"
    elif score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.1:
        return "LOW"
    else:
        return "MINIMAL"

def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for values
    """
    import scipy.stats as stats
    
    if not values:
        return {'lower': 0.0, 'upper': 0.0, 'mean': 0.0}
    
    values_array = np.array(values)
    mean = np.mean(values_array)
    std_err = stats.sem(values_array)
    
    interval = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=std_err)
    
    return {
        'lower': interval[0],
        'upper': interval[1],
        'mean': mean,
        'std_error': std_err
    }

def detect_outliers(data: List[float], method: str = 'iqr') -> Dict[str, Any]:
    """
    Detect outliers in numerical data
    """
    if not data:
        return {'outliers': [], 'outlier_indices': [], 'method': method}
    
    data_array = np.array(data)
    
    if method == 'iqr':
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (data_array < lower_bound) | (data_array > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data_array))
        outlier_mask = z_scores > 3
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    outliers = data_array[outlier_mask].tolist()
    outlier_indices = np.where(outlier_mask)[0].tolist()
    
    return {
        'outliers': outliers,
        'outlier_indices': outlier_indices,
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(data) * 100,
        'method': method
    }

def calculate_moving_average(values: List[float], window: int) -> List[float]:
    """
    Calculate moving average with specified window
    """
    if len(values) < window:
        return values
    
    return pd.Series(values).rolling(window=window).mean().tolist()[window-1:]

def detect_trend(values: List[float]) -> Dict[str, Any]:
    """
    Detect trend in time series data
    """
    if len(values) < 3:
        return {'trend': 'insufficient_data', 'slope': 0.0, 'r_squared': 0.0}
    
    x = np.arange(len(values))
    y = np.array(values)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Classify trend
    if abs(slope) < 0.01:
        trend = 'stable'
    elif slope > 0:
        trend = 'increasing'
    else:
        trend = 'decreasing'
    
    return {
        'trend': trend,
        'slope': slope,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'significance': 'significant' if p_value < 0.05 else 'not_significant'
    }

def calculate_entropy(values: List[Any]) -> float:
    """
    Calculate Shannon entropy of discrete values
    """
    if not values:
        return 0.0
    
    from collections import Counter
    import math
    
    counts = Counter(values)
    total = len(values)
    
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_gini_coefficient(values: List[float]) -> float:
    """
    Calculate Gini coefficient for measuring inequality
    """
    if not values:
        return 0.0
    
    # Sort values
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n

def format_number_with_suffix(number: float) -> str:
    """
    Format large numbers with K, M, B suffixes
    """
    if number < 1000:
        return f"{number:.1f}"
    elif number < 1000000:
        return f"{number/1000:.1f}K"
    elif number < 1000000000:
        return f"{number/1000000:.1f}M"
    else:
        return f"{number/1000000000:.1f}B"

def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """
    Convert datetime between timezones
    """
    from pytz import timezone
    
    from_timezone = timezone(from_tz)
    to_timezone = timezone(to_tz)
    
    # Localize if naive
    if dt.tzinfo is None:
        dt = from_timezone.localize(dt)
    
    return dt.astimezone(to_timezone)

def parse_date_range(date_range_str: str) -> Dict[str, datetime]:
    """
    Parse date range string into start and end dates
    """
    # Common date range formats
    now = datetime.now()
    
    if date_range_str == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif date_range_str == 'yesterday':
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif date_range_str == 'last_7_days':
        end = now
        start = now - timedelta(days=7)
    elif date_range_str == 'last_30_days':
        end = now
        start = now - timedelta(days=30)
    elif date_range_str == 'this_month':
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif date_range_str == 'last_month':
        last_month = now.replace(day=1) - timedelta(days=1)
        start = last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = last_month.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        # Try to parse custom format "YYYY-MM-DD to YYYY-MM-DD"
        try:
            start_str, end_str = date_range_str.split(' to ')
            start = datetime.strptime(start_str, '%Y-%m-%d')
            end = datetime.strptime(end_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date range format: {date_range_str}")
    
    return {'start': start, 'end': end}

def validate_data_consistency(data: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate data consistency based on rules
    """
    violations = []
    
    for rule in rules:
        rule_type = rule.get('type')
        
        if rule_type == 'required_field':
            field = rule['field']
            if field not in data or data[field] is None:
                violations.append(f"Required field '{field}' is missing")
        
        elif rule_type == 'range':
            field = rule['field']
            min_val = rule.get('min')
            max_val = rule.get('max')
            
            if field in data and data[field] is not None:
                value = data[field]
                if min_val is not None and value < min_val:
                    violations.append(f"Field '{field}' value {value} is below minimum {min_val}")
                if max_val is not None and value > max_val:
                    violations.append(f"Field '{field}' value {value} is above maximum {max_val}")
        
        elif rule_type == 'conditional':
            condition_field = rule['condition_field']
            condition_value = rule['condition_value']
            required_field = rule['required_field']
            
            if data.get(condition_field) == condition_value:
                if required_field not in data or data[required_field] is None:
                    violations.append(f"Field '{required_field}' is required when '{condition_field}' is '{condition_value}'")
    
    return {
        'is_valid': len(violations) == 0,
        'violations': violations
    }

def calculate_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate weighted score from multiple components
    """
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(scores.get(key, 0.0) * weight for key, weight in weights.items())
    return weighted_sum / total_weight

def generate_correlation_matrix(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate correlation matrix with insights
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Find highly correlated pairs
    high_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:  # High correlation threshold
                high_correlations.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'high_correlations': high_correlations,
        'mean_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    }

def create_summary_statistics(data: List[float]) -> Dict[str, float]:
    """
    Create comprehensive summary statistics
    """
    if not data:
        return {}
    
    data_array = np.array(data)
    
    return {
        'count': len(data),
        'mean': float(np.mean(data_array)),
        'median': float(np.median(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'q1': float(np.percentile(data_array, 25)),
        'q3': float(np.percentile(data_array, 75)),
        'iqr': float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
        'skewness': float(pd.Series(data).skew()),
        'kurtosis': float(pd.Series(data).kurtosis())
    }

class DataProcessor:
    """
    Utility class for data processing operations
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text data
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        return text
    
    @staticmethod
    def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize features to 0-1 range
        """
        if not features:
            return features
        
        values = list(features.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {key: 0.5 for key in features.keys()}
        
        return {
            key: (value - min_val) / (max_val - min_val)
            for key, value in features.items()
        }
    
    @staticmethod
    def handle_missing_values(data: Dict[str, Any], strategy: str = 'default') -> Dict[str, Any]:
        """
        Handle missing values in data
        """
        cleaned_data = data.copy()
        
        for key, value in cleaned_data.items():
            if value is None or (isinstance(value, str) and value.strip() == ''):
                if strategy == 'default':
                    # Use appropriate default based on key name
                    if 'count' in key.lower() or 'number' in key.lower():
                        cleaned_data[key] = 0
                    elif 'score' in key.lower() or 'ratio' in key.lower():
                        cleaned_data[key] = 0.0
                    elif 'bool' in key.lower() or key.lower().startswith('is_'):
                        cleaned_data[key] = False
                    else:
                        cleaned_data[key] = ""
                elif strategy == 'remove':
                    del cleaned_data[key]
        
        return cleaned_data