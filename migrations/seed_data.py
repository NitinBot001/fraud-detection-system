"""
Optional seed data for development and testing
"""
import sys
import os
import random
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database_config import SessionLocal
from app.models.database import *
from app.utils.logger import get_logger

logger = get_logger(__name__)

def create_sample_phone_numbers(db: SessionLocal, count: int = 100):
    """Create sample phone numbers for testing"""
    logger.info(f"Creating {count} sample phone numbers...")
    
    countries = [
        ('1', 'US'), ('44', 'UK'), ('91', 'IN'), ('86', 'CN'),
        ('49', 'DE'), ('33', 'FR'), ('39', 'IT'), ('7', 'RU')
    ]
    
    carriers = ['Verizon', 'AT&T', 'T-Mobile', 'Sprint', 'O2', 'Vodafone', 'Airtel']
    
    phone_numbers = []
    
    for i in range(count):
        country_code, country = random.choice(countries)
        
        # Generate random phone number
        if country_code == '1':  # US format
            number = f"+1{random.randint(200, 999)}{random.randint(200, 999)}{random.randint(1000, 9999)}"
        elif country_code == '44':  # UK format
            number = f"+44{random.randint(7000000000, 7999999999)}"
        else:  # Generic format
            number = f"+{country_code}{random.randint(1000000000, 9999999999)}"
        
        phone_record = PhoneNumber(
            number=number,
            country_code=country_code,
            is_mobile=random.choice([True, False]),
            carrier=random.choice(carriers) if random.random() > 0.3 else None,
            location=f"{country} Region {random.randint(1, 10)}",
            is_active=True
        )
        
        phone_numbers.append(phone_record)
    
    db.bulk_save_objects(phone_numbers)
    db.commit()
    
    logger.info(f"Created {count} sample phone numbers")
    return phone_numbers

def create_sample_fraud_reports(db: SessionLocal, phone_numbers: list, count: int = 200):
    """Create sample fraud reports"""
    logger.info(f"Creating {count} sample fraud reports...")
    
    fraud_types = ['SCAM_CALL', 'PHISHING', 'IDENTITY_THEFT', 'FINANCIAL_FRAUD', 
                  'ROBOCALL', 'SPAM', 'HARASSMENT', 'SPOOFING']
    severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    statuses = ['PENDING', 'VERIFIED', 'REJECTED']
    
    descriptions = [
        "Suspicious call attempting to get personal information",
        "Robocall advertising fake products",
        "Threatening call demanding money",
        "Impersonating bank representative",
        "Fake charity donation request",
        "Phishing attempt for credit card details",
        "Unwanted promotional calls",
        "Caller ID spoofing detected"
    ]
    
    reports = []
    
    for i in range(count):
        phone = random.choice(phone_numbers)
        
        report = FraudReport(
            phone_number_id=phone.id,
            fraud_type=random.choice(fraud_types),
            severity=random.choice(severities),
            description=random.choice(descriptions),
            reporter_ip=f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            reporter_location=f"City {random.randint(1, 100)}",
            evidence={'call_duration': random.randint(10, 300), 'time_of_day': random.randint(0, 23)},
            status=random.choice(statuses),
            confidence_score=random.uniform(0.3, 1.0),
            created_at=datetime.now() - timedelta(days=random.randint(0, 90))
        )
        
        reports.append(report)
    
    db.bulk_save_objects(reports)
    db.commit()
    
    logger.info(f"Created {count} sample fraud reports")
    return reports

def create_sample_network_connections(db: SessionLocal, phone_numbers: list, count: int = 300):
    """Create sample network connections"""
    logger.info(f"Creating {count} sample network connections...")
    
    connection_types = ['CALL', 'SMS', 'SHARED_DEVICE', 'SHARED_LOCATION', 'SOCIAL_LINK']
    
    connections = []
    used_pairs = set()
    
    for i in range(count):
        # Avoid duplicate connections
        attempts = 0
        while attempts < 10:
            source = random.choice(phone_numbers)
            target = random.choice(phone_numbers)
            
            if source.id != target.id:
                pair = tuple(sorted([source.id, target.id]))
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    break
            attempts += 1
        else:
            continue  # Skip if couldn't find unique pair
        
        connection = NetworkConnection(
            source_phone_id=source.id,
            target_phone_id=target.id,
            connection_type=random.choice(connection_types),
            strength=random.uniform(0.1, 1.0),
            frequency=random.randint(1, 50),
            last_interaction=datetime.now() - timedelta(days=random.randint(0, 30))
        )
        
        connections.append(connection)
    
    db.bulk_save_objects(connections)
    db.commit()
    
    logger.info(f"Created {len(connections)} sample network connections")
    return connections

def create_sample_geolocation_data(db: SessionLocal, phone_numbers: list, count: int = 500):
    """Create sample geolocation data"""
    logger.info(f"Creating {count} sample geolocation entries...")
    
    # Major cities coordinates
    cities = [
        (40.7128, -74.0060, 'US', 'NY', 'New York'),
        (34.0522, -118.2437, 'US', 'CA', 'Los Angeles'),
        (51.5074, -0.1278, 'UK', 'England', 'London'),
        (48.8566, 2.3522, 'FR', 'Ile-de-France', 'Paris'),
        (52.5200, 13.4050, 'DE', 'Berlin', 'Berlin'),
        (35.6762, 139.6503, 'JP', 'Tokyo', 'Tokyo'),
        (19.0760, 72.8777, 'IN', 'MH', 'Mumbai'),
        (-33.8688, 151.2093, 'AU', 'NSW', 'Sydney')
    ]
    
    geo_data = []
    
    for i in range(count):
        phone = random.choice(phone_numbers)
        lat, lon, country, state, city = random.choice(cities)
        
        # Add some random variation to coordinates
        lat += random.uniform(-0.1, 0.1)
        lon += random.uniform(-0.1, 0.1)
        
        geo_entry = GeolocationData(
            phone_number_id=phone.id,
            latitude=lat,
            longitude=lon,
            country=country,
            state=state,
            city=city,
            accuracy=random.uniform(5.0, 100.0),
            timestamp=datetime.now() - timedelta(hours=random.randint(0, 720))
        )
        
        geo_data.append(geo_entry)
    
    db.bulk_save_objects(geo_data)
    db.commit()
    
    logger.info(f"Created {count} sample geolocation entries")
    return geo_data

def create_sample_risk_scores(db: SessionLocal, phone_numbers: list):
    """Create sample risk scores"""
    logger.info("Creating sample risk scores...")
    
    risk_scores = []
    
    for phone in phone_numbers:
        # Generate realistic risk scores
        base_risk = random.uniform(0.0, 1.0)
        
        risk_score = RiskScore(
            phone_number_id=phone.id,
            overall_score=base_risk,
            fraud_probability=base_risk * random.uniform(0.8, 1.2),
            network_risk=random.uniform(0.0, 0.8),
            behavior_risk=random.uniform(0.0, 0.8),
            historical_risk=random.uniform(0.0, 0.9),
            geographic_risk=random.uniform(0.0, 0.6),
            anomaly_score=random.uniform(0.0, 1.0),
            clustering_label=random.randint(0, 5),
            prediction_confidence=random.uniform(0.6, 0.95)
        )
        
        risk_scores.append(risk_score)
    
    db.bulk_save_objects(risk_scores)
    db.commit()
    
    logger.info(f"Created risk scores for {len(phone_numbers)} phone numbers")
    return risk_scores

def create_sample_identity_links(db: SessionLocal, phone_numbers: list, count: int = 150):
    """Create sample identity links"""
    logger.info(f"Creating {count} sample identity links...")
    
    identity_types = ['AADHAR', 'EMAIL', 'DEVICE_ID', 'SSN', 'PASSPORT']
    verification_statuses = ['VERIFIED', 'PENDING', 'REJECTED', 'UNVERIFIED']
    
    identity_links = []
    
    for i in range(count):
        phone = random.choice(phone_numbers)
        identity_type = random.choice(identity_types)
        
        # Generate identity value based on type
        if identity_type == 'AADHAR':
            identity_value = f"{random.randint(100000000000, 999999999999)}"
        elif identity_type == 'EMAIL':
            identity_value = f"user{random.randint(1000, 9999)}@example.com"
        elif identity_type == 'DEVICE_ID':
            identity_value = f"device_{random.randint(100000, 999999)}"
        elif identity_type == 'SSN':
            identity_value = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        else:  # PASSPORT
            identity_value = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}{random.randint(1000000, 9999999)}"
        
        link = IdentityLink(
            phone_number_id=phone.id,
            identity_type=identity_type,
            identity_value=identity_value,
            verification_status=random.choice(verification_statuses),
            confidence_score=random.uniform(0.3, 1.0)
        )
        
        identity_links.append(link)
    
    db.bulk_save_objects(identity_links)
    db.commit()
    
    logger.info(f"Created {count} sample identity links")
    return identity_links

def seed_development_data():
    """Seed database with development data"""
    logger.info("Starting to seed development data...")
    
    db = SessionLocal()
    
    try:
        # Create sample data
        phone_numbers = create_sample_phone_numbers(db, 100)
        fraud_reports = create_sample_fraud_reports(db, phone_numbers, 200)
        network_connections = create_sample_network_connections(db, phone_numbers, 300)
        geo_data = create_sample_geolocation_data(db, phone_numbers, 500)
        risk_scores = create_sample_risk_scores(db, phone_numbers)
        identity_links = create_sample_identity_links(db, phone_numbers, 150)
        
        logger.info("Development data seeding completed successfully!")
        print("\n✅ Development data seeded successfully!")
        print(f"Created:")
        print(f"  - {len(phone_numbers)} phone numbers")
        print(f"  - {len(fraud_reports)} fraud reports")
        print(f"  - {len(network_connections)} network connections")
        print(f"  - {len(geo_data)} geolocation entries")
        print(f"  - {len(risk_scores)} risk scores")
        print(f"  - {len(identity_links)} identity links")
        
    except Exception as e:
        logger.error(f"Failed to seed development data: {str(e)}")
        db.rollback()
        print(f"\n❌ Failed to seed development data: {str(e)}")
    finally:
        db.close()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--confirm':
        seed_development_data()
    else:
        print("This will seed the database with sample data for development.")
        print("Run with --confirm flag to proceed: python seed_data.py --confirm")