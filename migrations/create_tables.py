"""
Database migration script to create all tables
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from config.database_config import Base, engine
from config.settings import Config
from app.models.database import *
from app.models.user_models import *
from app.utils.logger import get_logger
from app.utils.security import hash_password

logger = get_logger(__name__)

def create_all_tables():
    """Create all database tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {str(e)}")
        return False

def create_indexes():
    """Create additional database indexes for performance"""
    try:
        logger.info("Creating database indexes...")
        
        indexes = [
            # Phone number indexes
            "CREATE INDEX IF NOT EXISTS idx_phone_number_hash ON phone_numbers USING hash(number);",
            "CREATE INDEX IF NOT EXISTS idx_phone_created_at ON phone_numbers(created_at);",
            
            # Fraud report indexes
            "CREATE INDEX IF NOT EXISTS idx_fraud_report_phone_time ON fraud_reports(phone_number_id, created_at);",
            "CREATE INDEX IF NOT EXISTS idx_fraud_report_status ON fraud_reports(status);",
            "CREATE INDEX IF NOT EXISTS idx_fraud_report_severity_time ON fraud_reports(severity, created_at);",
            
            # Network connection indexes
            "CREATE INDEX IF NOT EXISTS idx_network_source_target ON network_connections(source_phone_id, target_phone_id);",
            "CREATE INDEX IF NOT EXISTS idx_network_strength ON network_connections(strength);",
            "CREATE INDEX IF NOT EXISTS idx_network_last_interaction ON network_connections(last_interaction);",
            
            # Risk score indexes
            "CREATE INDEX IF NOT EXISTS idx_risk_score_overall ON risk_scores(overall_score);",
            "CREATE INDEX IF NOT EXISTS idx_risk_score_phone ON risk_scores(phone_number_id);",
            "CREATE INDEX IF NOT EXISTS idx_risk_score_updated ON risk_scores(updated_at);",
            
            # Geolocation indexes
            "CREATE INDEX IF NOT EXISTS idx_geo_phone_time ON geolocation_data(phone_number_id, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_geo_coordinates ON geolocation_data(latitude, longitude);",
            "CREATE INDEX IF NOT EXISTS idx_geo_country_city ON geolocation_data(country, city);",
            
            # Behavior pattern indexes
            "CREATE INDEX IF NOT EXISTS idx_behavior_phone_type ON behavior_patterns(phone_number_id, pattern_type);",
            "CREATE INDEX IF NOT EXISTS idx_behavior_confidence ON behavior_patterns(confidence);",
            "CREATE INDEX IF NOT EXISTS idx_behavior_anomaly ON behavior_patterns(anomaly_detected);",
            
            # Identity link indexes
            "CREATE INDEX IF NOT EXISTS idx_identity_type_value ON identity_links(identity_type, identity_value);",
            "CREATE INDEX IF NOT EXISTS idx_identity_phone ON identity_links(phone_number_id);",
            "CREATE INDEX IF NOT EXISTS idx_identity_verification ON identity_links(verification_status);",
            
            # Audit log indexes
            "CREATE INDEX IF NOT EXISTS idx_audit_user_action ON audit_logs(user_id, action);",
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);",
            
            # User indexes
            "CREATE INDEX IF NOT EXISTS idx_user_username ON users(username);",
            "CREATE INDEX IF NOT EXISTS idx_user_email ON users(email);",
            "CREATE INDEX IF NOT EXISTS idx_user_active ON users(is_active);",
            
            # API key indexes
            "CREATE INDEX IF NOT EXISTS idx_api_key_hash ON api_keys(key_hash);",
            "CREATE INDEX IF NOT EXISTS idx_api_key_user ON api_keys(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_api_key_active ON api_keys(is_active);"
        ]
        
        with engine.connect() as connection:
            for index_sql in indexes:
                try:
                    connection.execute(text(index_sql))
                    logger.debug(f"Created index: {index_sql[:50]}...")
                except Exception as e:
                    logger.warning(f"Index creation failed: {str(e)}")
            
            connection.commit()
        
        logger.info("Database indexes created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create indexes: {str(e)}")
        return False

def create_admin_user():
    """Create default admin user"""
    try:
        from config.database_config import SessionLocal
        
        db = SessionLocal()
        
        # Check if admin user already exists
        admin_user = db.query(User).filter_by(username='admin').first()
        if admin_user:
            logger.info("Admin user already exists")
            return True
        
        # Create admin user
        admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')
        admin_user = User(
            username='admin',
            email='admin@fraud-detection.com',
            password_hash=hash_password(admin_password),
            is_active=True,
            is_admin=True,
            role='admin',
            full_name='System Administrator'
        )
        
        db.add(admin_user)
        db.commit()
        
        logger.info(f"Admin user created with username: admin")
        logger.warning(f"Default admin password: {admin_password}")
        logger.warning("Please change the admin password after first login!")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create admin user: {str(e)}")
        return False
    finally:
        db.close()

def create_sample_fraud_types():
    """Create sample fraud type reference data"""
    try:
        from config.database_config import SessionLocal
        
        fraud_types = [
            {
                'name': 'SCAM_CALL',
                'description': 'Fraudulent calls attempting to scam victims',
                'severity_weight': 0.8
            },
            {
                'name': 'PHISHING',
                'description': 'Attempts to steal personal information',
                'severity_weight': 0.9
            },
            {
                'name': 'IDENTITY_THEFT',
                'description': 'Using stolen identity information',
                'severity_weight': 0.95
            },
            {
                'name': 'FINANCIAL_FRAUD',
                'description': 'Fraudulent financial transactions',
                'severity_weight': 0.9
            },
            {
                'name': 'ROBOCALL',
                'description': 'Automated spam calls',
                'severity_weight': 0.3
            },
            {
                'name': 'SPAM',
                'description': 'Unwanted promotional calls',
                'severity_weight': 0.2
            },
            {
                'name': 'HARASSMENT',
                'description': 'Harassing or threatening calls',
                'severity_weight': 0.7
            },
            {
                'name': 'SPOOFING',
                'description': 'Caller ID spoofing',
                'severity_weight': 0.8
            }
        ]
        
        # Note: This would create a fraud_types reference table if implemented
        logger.info("Fraud types reference data prepared")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create fraud types: {str(e)}")
        return False

def setup_database_functions():
    """Create database functions and triggers"""
    try:
        logger.info("Setting up database functions...")
        
        functions = [
            # Function to update updated_at timestamp
            """
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            """,
            
            # Function to calculate phone number hash
            """
            CREATE OR REPLACE FUNCTION calculate_phone_hash(phone_number TEXT)
            RETURNS TEXT AS $$
            BEGIN
                RETURN encode(digest(phone_number, 'sha256'), 'hex');
            END;
            $$ language 'plpgsql';
            """
        ]
        
        triggers = [
            # Update timestamp triggers
            "CREATE TRIGGER update_phone_numbers_updated_at BEFORE UPDATE ON phone_numbers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
            "CREATE TRIGGER update_fraud_reports_updated_at BEFORE UPDATE ON fraud_reports FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
            "CREATE TRIGGER update_risk_scores_updated_at BEFORE UPDATE ON risk_scores FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
            "CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();"
        ]
        
        with engine.connect() as connection:
            # Create functions
            for function_sql in functions:
                try:
                    connection.execute(text(function_sql))
                except Exception as e:
                    logger.warning(f"Function creation failed: {str(e)}")
            
            # Create triggers
            for trigger_sql in triggers:
                try:
                    connection.execute(text(trigger_sql))
                except Exception as e:
                    logger.warning(f"Trigger creation failed: {str(e)}")
            
            connection.commit()
        
        logger.info("Database functions and triggers created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup database functions: {str(e)}")
        return False

def run_migrations():
    """Run all database migrations"""
    logger.info("Starting database migration...")
    
    success = True
    
    # Create tables
    if not create_all_tables():
        success = False
    
    # Create indexes
    if not create_indexes():
        success = False
    
    # Setup functions and triggers
    if not setup_database_functions():
        success = False
    
    # Create admin user
    if not create_admin_user():
        success = False
    
    # Create fraud types
    if not create_sample_fraud_types():
        success = False
    
    if success:
        logger.info("Database migration completed successfully!")
        print("\n✅ Database migration completed successfully!")
        print("\nNext steps:")
        print("1. Start the application: python main.py")
        print("2. Access the dashboard: http://localhost:5000/dashboard")
        print("3. Login with username: admin")
        print("4. Change the default admin password")
    else:
        logger.error("Database migration failed!")
        print("\n❌ Database migration failed! Check logs for details.")
    
    return success

if __name__ == '__main__':
    run_migrations()