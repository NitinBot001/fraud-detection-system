import bcrypt
import secrets
import hmac
import hashlib
import jwt
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from app.utils.logger import get_logger
from config.settings import Config

logger = get_logger(__name__)

class SecurityUtils:
    def __init__(self):
        self.encryption_key = self._get_or_generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def _get_or_generate_encryption_key(self) -> bytes:
        """
        Get or generate encryption key for sensitive data
        """
        # In production, store this securely
        key = Config.SECRET_KEY.encode()[:32]  # Use first 32 bytes
        key = key.ljust(32, b'0')  # Pad to 32 bytes if needed
        return base64.urlsafe_b64encode(key)

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt
    """
    try:
        salt = bcrypt.gensalt(rounds=Config.BCRYPT_LOG_ROUNDS)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except Exception as e:
        logger.error(f"Password hashing failed: {str(e)}")
        raise

def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against hash
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return False

def generate_api_key() -> str:
    """
    Generate secure API key
    """
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage
    """
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(api_key: str, hashed: str) -> bool:
    """
    Verify API key against hash
    """
    return hmac.compare_digest(hash_api_key(api_key), hashed)

def encrypt_sensitive_data(data: str) -> str:
    """
    Encrypt sensitive data
    """
    try:
        security = SecurityUtils()
        encrypted = security.cipher_suite.encrypt(data.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Data encryption failed: {str(e)}")
        raise

def decrypt_sensitive_data(encrypted_data: str) -> str:
    """
    Decrypt sensitive data
    """
    try:
        security = SecurityUtils()
        decrypted = security.cipher_suite.decrypt(encrypted_data.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Data decryption failed: {str(e)}")
        raise

def create_secure_token(data: Dict[str, Any], expires_in_hours: int = 24) -> str:
    """
    Create secure JWT token
    """
    try:
        payload = {
            **data,
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm='HS256')
        return token
    except Exception as e:
        logger.error(f"Token creation failed: {str(e)}")
        raise

def verify_secure_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode JWT token
    """
    try:
        payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        return None

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal
    """
    import re
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Remove leading dots and spaces
    sanitized = sanitized.lstrip('. ')
    # Limit length
    return sanitized[:255]

def validate_file_upload(file_data: bytes, allowed_types: list, max_size: int) -> Dict[str, Any]:
    """
    Validate file upload for security
    """
    import magic
    
    validation_result = {
        'valid': True,
        'errors': []
    }
    
    # Check file size
    if len(file_data) > max_size:
        validation_result['valid'] = False
        validation_result['errors'].append(f"File size exceeds maximum of {max_size} bytes")
    
    # Check file type using magic numbers
    try:
        file_type = magic.from_buffer(file_data, mime=True)
        if file_type not in allowed_types:
            validation_result['valid'] = False
            validation_result['errors'].append(f"File type {file_type} not allowed")
    except Exception as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Could not determine file type: {str(e)}")
    
    return validation_result

def generate_csrf_token() -> str:
    """
    Generate CSRF token
    """
    return secrets.token_urlsafe(32)

def verify_csrf_token(token: str, session_token: str) -> bool:
    """
    Verify CSRF token
    """
    return hmac.compare_digest(token, session_token)

def rate_limit_key(identifier: str, endpoint: str) -> str:
    """
    Generate rate limiting key
    """
    return f"rate_limit:{endpoint}:{identifier}"

def hash_phone_number(phone_number: str) -> str:
    """
    Hash phone number for privacy
    """
    return hashlib.sha256(phone_number.encode()).hexdigest()

def mask_phone_number(phone_number: str) -> str:
    """
    Mask phone number for display
    """
    if len(phone_number) <= 4:
        return "*" * len(phone_number)
    
    return phone_number[:2] + "*" * (len(phone_number) - 4) + phone_number[-2:]

def validate_input_length(data: str, min_length: int = 0, max_length: int = 1000) -> bool:
    """
    Validate input length
    """
    return min_length <= len(data) <= max_length

def escape_sql_like(value: str) -> str:
    """
    Escape SQL LIKE wildcards
    """
    return value.replace('%', r'\%').replace('_', r'\_')

import base64

class AuditLogger:
    """
    Security audit logging utility
    """
    
    def __init__(self):
        self.logger = get_logger('security_audit')
    
    def log_authentication(self, username: str, success: bool, ip_address: str, user_agent: str):
        """
        Log authentication attempt
        """
        self.logger.info("Authentication attempt", extra={
            'event_type': 'authentication',
            'username': username,
            'success': success,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_authorization(self, user_id: str, resource: str, action: str, success: bool):
        """
        Log authorization attempt
        """
        self.logger.info("Authorization attempt", extra={
            'event_type': 'authorization',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_data_access(self, user_id: str, data_type: str, record_id: str):
        """
        Log sensitive data access
        """
        self.logger.info("Data access", extra={
            'event_type': 'data_access',
            'user_id': user_id,
            'data_type': data_type,
            'record_id': record_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """
        Log security event
        """
        self.logger.warning("Security event", extra={
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })

class InputValidator:
    """
    Input validation utility
    """
    
    @staticmethod
    def validate_phone_number_format(phone: str) -> bool:
        """
        Validate phone number format
        """
        import re
        # Basic international phone number pattern
        pattern = r'^\+?[1-9]\d{1,14}$'
        return bool(re.match(pattern, re.sub(r'[\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE]', '', phone)))
    
    @staticmethod
    def validate_email_format(email: str) -> bool:
        """
        Validate email format
        """
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """
        Validate IP address
        """
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_json_structure(data: str, required_fields: list) -> Dict[str, Any]:
        """
        Validate JSON structure
        """
        try:
            parsed = json.loads(data)
            missing_fields = [field for field in required_fields if field not in parsed]
            
            return {
                'valid': len(missing_fields) == 0,
                'missing_fields': missing_fields,
                'parsed_data': parsed if len(missing_fields) == 0 else None
            }
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'error': f"Invalid JSON: {str(e)}"
            }
    
    @staticmethod
    def sanitize_html_input(html: str) -> str:
        """
        Sanitize HTML input
        """
        import html
        return html.escape(html)
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Validate date range
        """
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            if start > end:
                return {'valid': False, 'error': 'Start date must be before end date'}
            
            if (end - start).days > 365:
                return {'valid': False, 'error': 'Date range cannot exceed 365 days'}
            
            return {'valid': True, 'start_date': start, 'end_date': end}
            
        except ValueError as e:
            return {'valid': False, 'error': f'Invalid date format: {str(e)}'}

# Security middleware utilities
def get_client_ip(request) -> str:
    """
    Get client IP address from request
    """
    # Check for forwarded IPs (when behind proxy)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    return request.remote_addr

def detect_suspicious_activity(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect suspicious activity patterns
    """
    suspicious_indicators = []
    risk_score = 0.0
    
    # Check for SQL injection patterns
    sql_patterns = ['union', 'select', 'drop', 'insert', 'delete', '--', ';']
    for field, value in request_data.items():
        if isinstance(value, str):
            for pattern in sql_patterns:
                if pattern.lower() in value.lower():
                    suspicious_indicators.append(f"SQL injection pattern in {field}")
                    risk_score += 0.3
    
    # Check for XSS patterns
    xss_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
    for field, value in request_data.items():
        if isinstance(value, str):
            for pattern in xss_patterns:
                if pattern.lower() in value.lower():
                    suspicious_indicators.append(f"XSS pattern in {field}")
                    risk_score += 0.4
    
    # Check for path traversal
    if any('../' in str(value) or '..\\' in str(value) for value in request_data.values()):
        suspicious_indicators.append("Path traversal attempt")
        risk_score += 0.5
    
    return {
        'is_suspicious': risk_score > 0.5,
        'risk_score': min(1.0, risk_score),
        'indicators': suspicious_indicators
    }