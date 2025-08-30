import re
import phonenumbers
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.utils.logger import get_logger

logger = get_logger(__name__)

def validate_phone_number(phone_number: str) -> bool:
    """
    Validate phone number format
    """
    if not phone_number:
        return False
    
    try:
        # Clean the phone number
        cleaned = re.sub(r'[^\d+]', '', phone_number)
        
        # Parse using phonenumbers library
        parsed = phonenumbers.parse(cleaned, None)
        
        # Check if valid
        return phonenumbers.is_valid_number(parsed)
        
    except (phonenumbers.NumberParseException, Exception):
        # Fallback to regex validation
        pattern = r'^[\+]?[1-9]\d{1,14}$'
        return bool(re.match(pattern, cleaned))

def validate_fraud_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate fraud report data
    """
    errors = []
    
    # Required fields
    required_fields = ['phone_number', 'fraud_type']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"'{field}' is required")
    
    # Validate phone number
    if 'phone_number' in data:
        if not validate_phone_number(data['phone_number']):
            errors.append("Invalid phone number format")
    
    # Validate fraud type
    valid_fraud_types = [
        'SCAM_CALL', 'PHISHING', 'IDENTITY_THEFT', 'FINANCIAL_FRAUD',
        'ROBOCALL', 'SPAM', 'HARASSMENT', 'SPOOFING', 'OTHER'
    ]
    if 'fraud_type' in data and data['fraud_type'] not in valid_fraud_types:
        errors.append(f"Invalid fraud type. Must be one of: {', '.join(valid_fraud_types)}")
    
    # Validate severity
    if 'severity' in data:
        valid_severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        if data['severity'] not in valid_severities:
            errors.append(f"Invalid severity. Must be one of: {', '.join(valid_severities)}")
    
    # Validate confidence score
    if 'confidence_score' in data:
        try:
            score = float(data['confidence_score'])
            if not 0 <= score <= 1:
                errors.append("Confidence score must be between 0 and 1")
        except (ValueError, TypeError):
            errors.append("Confidence score must be a number")
    
    # Validate evidence
    if 'evidence' in data and not isinstance(data['evidence'], dict):
        errors.append("Evidence must be a JSON object")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_batch_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate batch analysis request
    """
    errors = []
    
    # Check phone_numbers field
    if 'phone_numbers' not in data:
        errors.append("'phone_numbers' field is required")
    elif not isinstance(data['phone_numbers'], list):
        errors.append("'phone_numbers' must be a list")
    elif len(data['phone_numbers']) == 0:
        errors.append("'phone_numbers' list cannot be empty")
    elif len(data['phone_numbers']) > 100:
        errors.append("Maximum 100 phone numbers allowed per batch")
    else:
        # Validate each phone number
        invalid_numbers = []
        for i, phone in enumerate(data['phone_numbers']):
            if not validate_phone_number(phone):
                invalid_numbers.append(f"Index {i}: {phone}")
        
        if invalid_numbers:
            errors.append(f"Invalid phone numbers: {', '.join(invalid_numbers)}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_user_registration(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate user registration data
    """
    errors = []
    
    # Required fields
    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"'{field}' is required")
    
    # Validate username
    if 'username' in data:
        username = data['username']
        if len(username) < 3:
            errors.append("Username must be at least 3 characters long")
        elif len(username) > 50:
            errors.append("Username must be less than 50 characters")
        elif not re.match(r'^[a-zA-Z0-9_]+$', username):
            errors.append("Username can only contain letters, numbers, and underscores")
    
    # Validate email
    if 'email' in data:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data['email']):
            errors.append("Invalid email format")
    
    # Validate password
    if 'password' in data:
        password = data['password']
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        elif not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        elif not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        elif not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        elif not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def sanitize_input(data: Any) -> Any:
    """
    Sanitize input data to prevent injection attacks
    """
    if isinstance(data, str):
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';]', '', data)
        return sanitized.strip()
    elif isinstance(data, dict):
        return {key: sanitize_input(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    else:
        return data