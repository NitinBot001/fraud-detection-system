from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from config.database_config import Base
import uuid

class BaseModel(Base):
    __abstract__ = True
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class PhoneNumber(BaseModel):
    __tablename__ = 'phone_numbers'
    
    number = Column(String(15), unique=True, nullable=False, index=True)
    country_code = Column(String(5), nullable=False)
    is_mobile = Column(Boolean, default=True)
    carrier = Column(String(100))
    location = Column(String(200))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    fraud_reports = relationship("FraudReport", back_populates="phone_number")
    risk_scores = relationship("RiskScore", back_populates="phone_number")
    behavior_patterns = relationship("BehaviorPattern", back_populates="phone_number")

class FraudReport(BaseModel):
    __tablename__ = 'fraud_reports'
    
    phone_number_id = Column(String, ForeignKey('phone_numbers.id'), nullable=False)
    fraud_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    description = Column(Text)
    reporter_ip = Column(String(45))
    reporter_location = Column(String(200))
    evidence = Column(JSON)
    status = Column(String(20), default='PENDING')  # PENDING, VERIFIED, REJECTED
    confidence_score = Column(Float, default=0.0)
    
    # Relationships
    phone_number = relationship("PhoneNumber", back_populates="fraud_reports")
    
    __table_args__ = (
        Index('idx_fraud_reports_phone_type', 'phone_number_id', 'fraud_type'),
        Index('idx_fraud_reports_severity', 'severity'),
    )

class RiskScore(BaseModel):
    __tablename__ = 'risk_scores'
    
    phone_number_id = Column(String, ForeignKey('phone_numbers.id'), nullable=False)
    overall_score = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    network_risk = Column(Float, default=0.0)
    behavior_risk = Column(Float, default=0.0)
    historical_risk = Column(Float, default=0.0)
    geographic_risk = Column(Float, default=0.0)
    
    # ML Model outputs
    anomaly_score = Column(Float, default=0.0)
    clustering_label = Column(Integer)
    prediction_confidence = Column(Float, default=0.0)
    
    # Relationships
    phone_number = relationship("PhoneNumber", back_populates="risk_scores")
    
    __table_args__ = (
        Index('idx_risk_scores_overall', 'overall_score'),
        Index('idx_risk_scores_fraud_prob', 'fraud_probability'),
    )

class NetworkConnection(BaseModel):
    __tablename__ = 'network_connections'
    
    source_phone_id = Column(String, ForeignKey('phone_numbers.id'), nullable=False)
    target_phone_id = Column(String, ForeignKey('phone_numbers.id'), nullable=False)
    connection_type = Column(String(50), nullable=False)  # CALL, SMS, SHARED_DEVICE, etc.
    strength = Column(Float, default=1.0)
    frequency = Column(Integer, default=1)
    last_interaction = Column(DateTime)
    
    __table_args__ = (
        Index('idx_network_source_target', 'source_phone_id', 'target_phone_id'),
        Index('idx_network_strength', 'strength'),
    )

class BehaviorPattern(BaseModel):
    __tablename__ = 'behavior_patterns'
    
    phone_number_id = Column(String, ForeignKey('phone_numbers.id'), nullable=False)
    pattern_type = Column(String(50), nullable=False)
    pattern_data = Column(JSON, nullable=False)
    confidence = Column(Float, default=0.0)
    anomaly_detected = Column(Boolean, default=False)
    
    # Relationships
    phone_number = relationship("PhoneNumber", back_populates="behavior_patterns")
    
    __table_args__ = (
        Index('idx_behavior_phone_type', 'phone_number_id', 'pattern_type'),
    )

class IdentityLink(BaseModel):
    __tablename__ = 'identity_links'
    
    phone_number_id = Column(String, ForeignKey('phone_numbers.id'), nullable=False)
    identity_type = Column(String(50), nullable=False)  # AADHAR, EMAIL, DEVICE_ID, etc.
    identity_value = Column(String(255), nullable=False)
    verification_status = Column(String(20), default='UNVERIFIED')
    confidence_score = Column(Float, default=0.0)
    
    __table_args__ = (
        Index('idx_identity_type_value', 'identity_type', 'identity_value'),
        Index('idx_identity_phone', 'phone_number_id'),
    )

class GeolocationData(BaseModel):
    __tablename__ = 'geolocation_data'
    
    phone_number_id = Column(String, ForeignKey('phone_numbers.id'), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    country = Column(String(100))
    state = Column(String(100))
    city = Column(String(100))
    accuracy = Column(Float)
    timestamp = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_geo_phone_time', 'phone_number_id', 'timestamp'),
        Index('idx_geo_location', 'latitude', 'longitude'),
    )

class MLModel(BaseModel):
    __tablename__ = 'ml_models'
    
    model_name = Column(String(100), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)  # ANOMALY, CLUSTERING, CLASSIFICATION
    version = Column(String(20), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_data_size = Column(Integer)
    last_trained = Column(DateTime)
    is_active = Column(Boolean, default=True)
    model_path = Column(String(500))
    hyperparameters = Column(JSON)
    
    __table_args__ = (
        Index('idx_model_name_version', 'model_name', 'version'),
    )

class AuditLog(BaseModel):
    __tablename__ = 'audit_logs'
    
    user_id = Column(String)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    details = Column(JSON)
    
    __table_args__ = (
        Index('idx_audit_user_action', 'user_id', 'action'),
        Index('idx_audit_timestamp', 'created_at'),
    )