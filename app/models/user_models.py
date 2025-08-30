from sqlalchemy import Column, String, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from flask_login import UserMixin
from app.models.database import BaseModel

class User(BaseModel, UserMixin):
    __tablename__ = 'users'
    
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    role = Column(String(50), default='analyst')
    
    # Profile information
    full_name = Column(String(200))
    organization = Column(String(200))
    phone = Column(String(20))
    
    # Security
    last_login = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime)
    
    # Preferences
    notification_preferences = Column(JSON, default={})
    dashboard_config = Column(JSON, default={})
    
    def get_id(self):
        return str(self.id)

class APIKey(BaseModel):
    __tablename__ = 'api_keys'
    
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    key_name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False)
    permissions = Column(JSON, default=[])
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    rate_limit = Column(Integer, default=1000)  # requests per hour
    
    __table_args__ = (
        Index('idx_api_key_hash', 'key_hash'),
        Index('idx_api_key_user', 'user_id'),
    )