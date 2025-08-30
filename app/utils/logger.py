import logging
import logging.handlers
import sys
import json
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import structlog
from config.settings import Config

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)

class SecurityLogFilter(logging.Filter):
    """
    Filter for security-related logs
    """
    
    def filter(self, record):
        # Only pass security-related logs
        security_keywords = ['authentication', 'authorization', 'fraud', 'security', 'attack']
        message = record.getMessage().lower()
        
        return any(keyword in message for keyword in security_keywords)

def setup_logging():
    """
    Set up application logging configuration
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for application logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "application.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Security logs
    security_handler = logging.handlers.RotatingFileHandler(
        log_dir / "security.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=10
    )
    security_handler.setLevel(logging.WARNING)
    security_handler.setFormatter(JSONFormatter())
    security_handler.addFilter(SecurityLogFilter())
    root_logger.addHandler(security_handler)
    
    # Error logs
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)
    
    # Suppress some noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get structured logger instance
    """
    return structlog.get_logger(name)

class AuditLogger:
    """
    Specialized logger for audit trails
    """
    
    def __init__(self):
        self.logger = get_logger('audit')
        
        # Create audit log file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        audit_handler = logging.handlers.RotatingFileHandler(
            log_dir / "audit.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=20
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(JSONFormatter())
        
        # Add handler to audit logger
        audit_logger = logging.getLogger('audit')
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
    
    def log_user_action(self, user_id: str, action: str, resource: str = None, 
                       details: Dict[str, Any] = None, ip_address: str = None):
        """
        Log user action for audit trail
        """
        self.logger.info(
            "User action",
            user_id=user_id,
            action=action,
            resource=resource,
            details=details or {},
            ip_address=ip_address,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_data_access(self, user_id: str, data_type: str, record_id: str = None,
                       operation: str = 'read', sensitive: bool = False):
        """
        Log data access for compliance
        """
        self.logger.info(
            "Data access",
            user_id=user_id,
            data_type=data_type,
            record_id=record_id,
            operation=operation,
            sensitive=sensitive,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_security_event(self, event_type: str, severity: str, description: str,
                          source_ip: str = None, user_id: str = None, 
                          additional_data: Dict[str, Any] = None):
        """
        Log security events
        """
        self.logger.warning(
            "Security event",
            event_type=event_type,
            severity=severity,
            description=description,
            source_ip=source_ip,
            user_id=user_id,
            additional_data=additional_data or {},
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_system_event(self, event_type: str, component: str, status: str,
                        message: str, metrics: Dict[str, Any] = None):
        """
        Log system events and metrics
        """
        self.logger.info(
            "System event",
            event_type=event_type,
            component=component,
            status=status,
            message=message,
            metrics=metrics or {},
            timestamp=datetime.utcnow().isoformat()
        )

class PerformanceLogger:
    """
    Logger for performance monitoring
    """
    
    def __init__(self):
        self.logger = get_logger('performance')
    
    def log_api_performance(self, endpoint: str, method: str, response_time: float,
                           status_code: int, user_id: str = None):
        """
        Log API performance metrics
        """
        self.logger.info(
            "API performance",
            endpoint=endpoint,
            method=method,
            response_time_ms=response_time * 1000,
            status_code=status_code,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_ml_performance(self, model_name: str, operation: str, duration: float,
                          input_size: int = None, accuracy: float = None):
        """
        Log ML model performance
        """
        self.logger.info(
            "ML performance",
            model_name=model_name,
            operation=operation,
            duration_ms=duration * 1000,
            input_size=input_size,
            accuracy=accuracy,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_database_performance(self, query_type: str, table: str, duration: float,
                               rows_affected: int = None):
        """
        Log database performance
        """
        self.logger.info(
            "Database performance",
            query_type=query_type,
            table=table,
            duration_ms=duration * 1000,
            rows_affected=rows_affected,
            timestamp=datetime.utcnow().isoformat()
        )

class FraudLogger:
    """
    Specialized logger for fraud detection events
    """
    
    def __init__(self):
        self.logger = get_logger('fraud')
        self.audit_logger = AuditLogger()
    
    def log_fraud_detection(self, phone_number: str, risk_score: float, 
                           risk_level: str, detected_patterns: list,
                           processing_time: float, user_id: str = None):
        """
        Log fraud detection results
        """
        # Mask phone number for privacy
        masked_phone = self._mask_phone_number(phone_number)
        
        self.logger.info(
            "Fraud detection",
            phone_number_hash=self._hash_phone_number(phone_number),
            masked_phone=masked_phone,
            risk_score=risk_score,
            risk_level=risk_level,
            detected_patterns=detected_patterns,
            processing_time_ms=processing_time * 1000,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Also log to audit trail if high risk
        if risk_score > 0.7:
            self.audit_logger.log_security_event(
                event_type='HIGH_RISK_DETECTION',
                severity='HIGH',
                description=f'High risk phone number detected: {masked_phone}',
                user_id=user_id,
                additional_data={
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'patterns': detected_patterns
                }
            )
    
    def log_fraud_report(self, phone_number: str, fraud_type: str, severity: str,
                        reporter_ip: str = None, user_id: str = None):
        """
        Log fraud report submission
        """
        self.logger.info(
            "Fraud report",
            phone_number_hash=self._hash_phone_number(phone_number),
            masked_phone=self._mask_phone_number(phone_number),
            fraud_type=fraud_type,
            severity=severity,
            reporter_ip=reporter_ip,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_model_prediction(self, model_name: str, phone_number: str,
                           prediction: float, confidence: float, features_used: int):
        """
        Log ML model predictions
        """
        self.logger.debug(
            "Model prediction",
            model_name=model_name,
            phone_number_hash=self._hash_phone_number(phone_number),
            prediction=prediction,
            confidence=confidence,
            features_used=features_used,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _mask_phone_number(self, phone_number: str) -> str:
        """
        Mask phone number for logging
        """
        if len(phone_number) <= 4:
            return "*" * len(phone_number)
        
        return phone_number[:2] + "*" * (len(phone_number) - 4) + phone_number[-2:]
    
    def _hash_phone_number(self, phone_number: str) -> str:
        """
        Hash phone number for consistent identification without revealing PII
        """
        import hashlib
        return hashlib.sha256(phone_number.encode()).hexdigest()[:16]

# Context manager for performance logging
class PerformanceTimer:
    """
    Context manager for timing operations
    """
    
    def __init__(self, operation_name: str, logger: Optional[PerformanceLogger] = None):
        self.operation_name = operation_name
        self.logger = logger or PerformanceLogger()
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.logger.info(
                f"Operation completed: {self.operation_name}",
                operation=self.operation_name,
                duration_ms=duration * 1000,
                status='success'
            )
        else:
            self.logger.logger.error(
                f"Operation failed: {self.operation_name}",
                operation=self.operation_name,
                duration_ms=duration * 1000,
                status='failed',
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None
            )
    
    def get_duration(self) -> float:
        """
        Get operation duration in seconds
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

# Initialize logging when module is imported
setup_logging()