from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import asyncio
import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import PhoneNumber, FraudReport, User, AuditLog
from app.core.fraud_detector import FraudDetector
from app.ml.anomaly_detector import AnomalyDetector
from app.ml.fraud_predictor import FraudPredictor
from app.ml.feature_engineer import FeatureEngineer
from app.api.validators import validate_phone_number, validate_fraud_report
from app.services.notification_service import NotificationService
from app.utils.logger import get_logger
from app.utils.security import hash_password, verify_password
from config.database_config import get_db

# Initialize blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

logger = get_logger(__name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@api_bp.route('/auth/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    """User authentication endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Get database session
        db = next(get_db())
        
        # Find user
        user = db.query(User).filter_by(username=username, is_active=True).first()
        
        if not user or not verify_password(password, user.password_hash):
            # Log failed login attempt
            audit_log = AuditLog(
                user_id=username,
                action='LOGIN_FAILED',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                details={'reason': 'Invalid credentials'}
            )
            db.add(audit_log)
            db.commit()
            
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Create access token
        access_token = create_access_token(
            identity=user.id,
            additional_claims={
                'username': user.username,
                'role': user.role,
                'is_admin': user.is_admin
            }
        )
        
        # Update last login
        user.last_login = datetime.now()
        user.failed_login_attempts = 0
        
        # Log successful login
        audit_log = AuditLog(
            user_id=user.id,
            action='LOGIN_SUCCESS',
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string
        )
        db.add(audit_log)
        db.commit()
        
        return jsonify({
            'access_token': access_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'role': user.role,
                'is_admin': user.is_admin
            }
        })
        
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return jsonify({'error': 'Authentication failed'}), 500

@api_bp.route('/fraud/analyze', methods=['POST'])
@jwt_required()
@limiter.limit("100 per hour")
async def analyze_fraud():
    """Comprehensive fraud analysis endpoint"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        deep_analysis = data.get('deep_analysis', True)
        
        # Validate phone number
        if not validate_phone_number(phone_number):
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        # Get database session
        db = next(get_db())
        
        # Initialize fraud detector
        fraud_detector = FraudDetector(db)
        
        # Run analysis
        result = await fraud_detector.analyze_phone_number(phone_number, deep_analysis)
        
        # Log analysis request
        user_id = get_jwt_identity()
        audit_log = AuditLog(
            user_id=user_id,
            action='FRAUD_ANALYSIS',
            resource_type='PHONE_NUMBER',
            resource_id=phone_number,
            ip_address=request.remote_addr,
            details={
                'deep_analysis': deep_analysis,
                'risk_score': result.risk_score,
                'risk_level': result.risk_level
            }
        )
        db.add(audit_log)
        db.commit()
        
        # Convert result to dict for JSON response
        response_data = {
            'phone_number': result.phone_number,
            'risk_score': result.risk_score,
            'fraud_probability': result.fraud_probability,
            'risk_level': result.risk_level,
            'confidence': result.confidence,
            'detected_patterns': result.detected_patterns,
            'network_risk': result.network_risk,
            'behavioral_anomalies': result.behavioral_anomalies,
            'recommendations': result.recommendations,
            'evidence': result.evidence,
            'processing_time': result.processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Fraud analysis failed: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@api_bp.route('/fraud/report', methods=['POST'])
@jwt_required()
@limiter.limit("50 per hour")
async def submit_fraud_report():
    """Submit fraud report endpoint"""
    try:
        data = request.get_json()
        
        # Validate input
        validation_result = validate_fraud_report(data)
        if not validation_result['valid']:
            return jsonify({'error': validation_result['errors']}), 400
        
        # Get database session
        db = next(get_db())
        
        # Get or create phone number record
        phone_number = data['phone_number']
        phone_record = db.query(PhoneNumber).filter_by(number=phone_number).first()
        
        if not phone_record:
            phone_record = PhoneNumber(
                number=phone_number,
                country_code=data.get('country_code', ''),
                is_mobile=data.get('is_mobile', True)
            )
            db.add(phone_record)
            db.flush()
        
        # Create fraud report
        fraud_report = FraudReport(
            phone_number_id=phone_record.id,
            fraud_type=data['fraud_type'],
            severity=data.get('severity', 'MEDIUM'),
            description=data.get('description', ''),
            reporter_ip=request.remote_addr,
            reporter_location=data.get('reporter_location'),
            evidence=data.get('evidence', {}),
            confidence_score=data.get('confidence_score', 0.5)
        )
        
        db.add(fraud_report)
        
        # Log report submission
        user_id = get_jwt_identity()
        audit_log = AuditLog(
            user_id=user_id,
            action='FRAUD_REPORT_SUBMITTED',
            resource_type='FRAUD_REPORT',
            resource_id=fraud_report.id,
            ip_address=request.remote_addr,
            details={
                'phone_number': phone_number,
                'fraud_type': data['fraud_type'],
                'severity': data.get('severity', 'MEDIUM')
            }
        )
        db.add(audit_log)
        db.commit()
        
        # Trigger real-time analysis if high severity
        if data.get('severity') in ['HIGH', 'CRITICAL']:
            # Send notification to administrators
            notification_service = NotificationService()
            await notification_service.send_alert(
                f"High severity fraud report for {phone_number}",
                {
                    'phone_number': phone_number,
                    'fraud_type': data['fraud_type'],
                    'severity': data['severity'],
                    'report_id': fraud_report.id
                }
            )
        
        return jsonify({
            'report_id': fraud_report.id,
            'status': 'submitted',
            'message': 'Fraud report submitted successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Fraud report submission failed: {str(e)}")
        return jsonify({'error': 'Report submission failed'}), 500

@api_bp.route('/fraud/batch-analyze', methods=['POST'])
@jwt_required()
@limiter.limit("10 per hour")
async def batch_analyze():
    """Batch fraud analysis endpoint"""
    try:
        data = request.get_json()
        phone_numbers = data.get('phone_numbers', [])
        
        if not phone_numbers or len(phone_numbers) > 100:
            return jsonify({'error': 'Invalid phone numbers list (max 100)'}), 400
        
        # Validate all phone numbers
        invalid_numbers = [num for num in phone_numbers if not validate_phone_number(num)]
        if invalid_numbers:
            return jsonify({'error': f'Invalid phone numbers: {invalid_numbers}'}), 400
        
        # Get database session
        db = next(get_db())
        
        # Initialize components
        fraud_detector = FraudDetector(db)
        feature_engineer = FeatureEngineer(db)
        fraud_predictor = FraudPredictor()
        
        # Process batch
        results = []
        
        for phone_number in phone_numbers:
            try:
                # Quick analysis for batch processing
                result = await fraud_detector.analyze_phone_number(phone_number, deep_analysis=False)
                
                results.append({
                    'phone_number': phone_number,
                    'risk_score': result.risk_score,
                    'risk_level': result.risk_level,
                    'fraud_probability': result.fraud_probability,
                    'confidence': result.confidence
                })
                
            except Exception as e:
                logger.warning(f"Analysis failed for {phone_number}: {str(e)}")
                results.append({
                    'phone_number': phone_number,
                    'error': 'Analysis failed',
                    'risk_score': 0.0,
                    'risk_level': 'UNKNOWN'
                })
        
        # Log batch analysis
        user_id = get_jwt_identity()
        audit_log = AuditLog(
            user_id=user_id,
            action='BATCH_ANALYSIS',
            ip_address=request.remote_addr,
            details={
                'phone_count': len(phone_numbers),
                'successful_analyses': len([r for r in results if 'error' not in r])
            }
        )
        db.add(audit_log)
        db.commit()
        
        return jsonify({
            'results': results,
            'total_analyzed': len(phone_numbers),
            'successful': len([r for r in results if 'error' not in r]),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        return jsonify({'error': 'Batch analysis failed'}), 500

@api_bp.route('/fraud/network-analysis', methods=['POST'])
@jwt_required()
@limiter.limit("20 per hour")
async def network_analysis():
    """Network analysis endpoint"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        depth = data.get('depth', 2)
        
        if not validate_phone_number(phone_number):
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        if depth > 3:
            return jsonify({'error': 'Maximum depth is 3'}), 400
        
        # Get database session
        db = next(get_db())
        
        # Get phone record
        phone_record = db.query(PhoneNumber).filter_by(number=phone_number).first()
        if not phone_record:
            return jsonify({'error': 'Phone number not found'}), 404
        
        # Initialize network analyzer
        from app.core.network_analyzer import NetworkAnalyzer
        network_analyzer = NetworkAnalyzer(db)
        
        # Run network analysis
        result = await network_analyzer.analyze_phone_network(phone_record)
        
        # Log network analysis
        user_id = get_jwt_identity()
        audit_log = AuditLog(
            user_id=user_id,
            action='NETWORK_ANALYSIS',
            resource_type='PHONE_NUMBER',
            resource_id=phone_number,
            ip_address=request.remote_addr,
            details={
                'depth': depth,
                'network_size': result.get('network_size', 0),
                'network_risk': result.get('network_risk', 0.0)
            }
        )
        db.add(audit_log)
        db.commit()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Network analysis failed: {str(e)}")
        return jsonify({'error': 'Network analysis failed'}), 500

@api_bp.route('/fraud/reports', methods=['GET'])
@jwt_required()
@limiter.limit("200 per hour")
def get_fraud_reports():
    """Get fraud reports with filtering and pagination"""
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 50, type=int), 100)
        phone_number = request.args.get('phone_number')
        fraud_type = request.args.get('fraud_type')
        severity = request.args.get('severity')
        status = request.args.get('status')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Get database session
        db = next(get_db())
        
        # Build query
        query = db.query(FraudReport)
        
        # Apply filters
        if phone_number:
            phone_record = db.query(PhoneNumber).filter_by(number=phone_number).first()
            if phone_record:
                query = query.filter(FraudReport.phone_number_id == phone_record.id)
            else:
                # No results if phone number doesn't exist
                return jsonify({
                    'reports': [],
                    'total': 0,
                    'page': page,
                    'per_page': per_page,
                    'pages': 0
                })
        
        if fraud_type:
            query = query.filter(FraudReport.fraud_type == fraud_type)
        
        if severity:
            query = query.filter(FraudReport.severity == severity)
        
        if status:
            query = query.filter(FraudReport.status == status)
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                query = query.filter(FraudReport.created_at >= start_dt)
            except ValueError:
                return jsonify({'error': 'Invalid start_date format'}), 400
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
                query = query.filter(FraudReport.created_at <= end_dt)
            except ValueError:
                return jsonify({'error': 'Invalid end_date format'}), 400
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        reports = query.order_by(FraudReport.created_at.desc()).offset(
            (page - 1) * per_page
        ).limit(per_page).all()
        
        # Format results
        formatted_reports = []
        for report in reports:
            phone_record = db.query(PhoneNumber).filter_by(id=report.phone_number_id).first()
            
            formatted_reports.append({
                'id': report.id,
                'phone_number': phone_record.number if phone_record else None,
                'fraud_type': report.fraud_type,
                'severity': report.severity,
                'status': report.status,
                'description': report.description,
                'confidence_score': report.confidence_score,
                'created_at': report.created_at.isoformat(),
                'reporter_location': report.reporter_location
            })
        
        # Calculate pagination info
        pages = (total + per_page - 1) // per_page
        
        return jsonify({
            'reports': formatted_reports,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': pages
        })
        
    except Exception as e:
        logger.error(f"Get fraud reports failed: {str(e)}")
        return jsonify({'error': 'Failed to retrieve reports'}), 500

@api_bp.route('/fraud/statistics', methods=['GET'])
@jwt_required()
@limiter.limit("100 per hour")
def get_fraud_statistics():
    """Get fraud statistics and analytics"""
    try:
        # Get database session
        db = next(get_db())
        
        # Time period for statistics
        days = request.args.get('days', 30, type=int)
        start_date = datetime.now() - timedelta(days=days)
        
        # Basic statistics
        total_reports = db.query(FraudReport).filter(
            FraudReport.created_at >= start_date
        ).count()
        
        verified_reports = db.query(FraudReport).filter(
            FraudReport.created_at >= start_date,
            FraudReport.status == 'VERIFIED'
        ).count()
        
        unique_numbers = db.query(FraudReport.phone_number_id).filter(
            FraudReport.created_at >= start_date
        ).distinct().count()
        
        # Fraud type distribution
        fraud_types = db.query(
            FraudReport.fraud_type, 
            db.func.count(FraudReport.id).label('count')
        ).filter(
            FraudReport.created_at >= start_date
        ).group_by(FraudReport.fraud_type).all()
        
        fraud_type_dist = {ft.fraud_type: ft.count for ft in fraud_types}
        
        # Severity distribution
        severities = db.query(
            FraudReport.severity,
            db.func.count(FraudReport.id).label('count')
        ).filter(
            FraudReport.created_at >= start_date
        ).group_by(FraudReport.severity).all()
        
        severity_dist = {s.severity: s.count for s in severities}
        
        # Daily trends
        daily_reports = db.query(
            db.func.date(FraudReport.created_at).label('date'),
            db.func.count(FraudReport.id).label('count')
        ).filter(
            FraudReport.created_at >= start_date
        ).group_by(db.func.date(FraudReport.created_at)).all()
        
        daily_trend = [
            {'date': dr.date.isoformat(), 'count': dr.count} 
            for dr in daily_reports
        ]
        
        # Top reported numbers
        top_numbers = db.query(
            FraudReport.phone_number_id,
            db.func.count(FraudReport.id).label('report_count')
        ).filter(
            FraudReport.created_at >= start_date
        ).group_by(FraudReport.phone_number_id).order_by(
            db.func.count(FraudReport.id).desc()
        ).limit(10).all()
        
        top_numbers_formatted = []
        for tn in top_numbers:
            phone_record = db.query(PhoneNumber).filter_by(id=tn.phone_number_id).first()
            if phone_record:
                top_numbers_formatted.append({
                    'phone_number': phone_record.number,
                    'report_count': tn.report_count
                })
        
        statistics = {
            'period_days': days,
            'total_reports': total_reports,
            'verified_reports': verified_reports,
            'unique_numbers_reported': unique_numbers,
            'verification_rate': verified_reports / total_reports if total_reports > 0 else 0,
            'fraud_type_distribution': fraud_type_dist,
            'severity_distribution': severity_dist,
            'daily_trend': daily_trend,
            'top_reported_numbers': top_numbers_formatted,
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(statistics)
        
    except Exception as e:
        logger.error(f"Get fraud statistics failed: {str(e)}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

@api_bp.route('/ml/train', methods=['POST'])
@jwt_required()
@limiter.limit("5 per day")
async def train_models():
    """Train ML models endpoint (admin only)"""
    try:
        # Check admin permissions
        user_id = get_jwt_identity()
        db = next(get_db())
        user = db.query(User).filter_by(id=user_id).first()
        
        if not user or not user.is_admin:
            return jsonify({'error': 'Admin privileges required'}), 403
        
        data = request.get_json()
        model_types = data.get('model_types', ['fraud_predictor', 'anomaly_detector'])
        
        results = {}
        
        # Train fraud prediction models
        if 'fraud_predictor' in model_types:
            try:
                fraud_predictor = FraudPredictor()
                
                # Get training data (this would be more sophisticated in practice)
                training_data = await _get_training_data(db)
                
                if len(training_data) >= 100:  # Minimum training samples
                    metrics = await fraud_predictor.train_models(training_data)
                    results['fraud_predictor'] = {
                        'status': 'success',
                        'metrics': metrics,
                        'training_samples': len(training_data)
                    }
                else:
                    results['fraud_predictor'] = {
                        'status': 'skipped',
                        'reason': 'Insufficient training data'
                    }
                    
            except Exception as e:
                results['fraud_predictor'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Train anomaly detection models
        if 'anomaly_detector' in model_types:
            try:
                anomaly_detector = AnomalyDetector()
                
                # Get feature data for training
                feature_data = await _get_feature_data(db)
                
                if len(feature_data) >= 100:
                    metrics = await anomaly_detector.train_models(feature_data)
                    results['anomaly_detector'] = {
                        'status': 'success',
                        'metrics': metrics,
                        'training_samples': len(feature_data)
                    }
                else:
                    results['anomaly_detector'] = {
                        'status': 'skipped',
                        'reason': 'Insufficient training data'
                    }
                    
            except Exception as e:
                results['anomaly_detector'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Log training request
        audit_log = AuditLog(
            user_id=user_id,
            action='MODEL_TRAINING',
            ip_address=request.remote_addr,
            details={
                'model_types': model_types,
                'results': results
            }
        )
        db.add(audit_log)
        db.commit()
        
        return jsonify({
            'training_results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return jsonify({'error': 'Model training failed'}), 500

async def _get_training_data(db: Session) -> pd.DataFrame:
    """Get training data for fraud prediction models"""
    # This would implement sophisticated training data collection
    # For now, return a basic dataset
    
    import pandas as pd
    
    # Get phone numbers with fraud reports
    fraud_phones = db.query(PhoneNumber).join(FraudReport).distinct().all()
    
    # Get phone numbers without fraud reports (sample)
    clean_phones = db.query(PhoneNumber).filter(
        ~PhoneNumber.id.in_(
            db.query(FraudReport.phone_number_id).distinct()
        )
    ).limit(len(fraud_phones) * 2).all()  # 2:1 ratio clean to fraud
    
    # Create training dataset
    training_data = []
    
    feature_engineer = FeatureEngineer(db)
    
    # Add fraud cases
    for phone in fraud_phones:
        features = await feature_engineer.engineer_comprehensive_features(phone)
        features['is_fraud'] = 1
        training_data.append(features)
    
    # Add clean cases
    for phone in clean_phones:
        features = await feature_engineer.engineer_comprehensive_features(phone)
        features['is_fraud'] = 0
        training_data.append(features)
    
    return pd.DataFrame(training_data)

async def _get_feature_data(db: Session) -> pd.DataFrame:
    """Get feature data for anomaly detection training"""
    # Similar to training data but focused on feature extraction
    phones = db.query(PhoneNumber).limit(1000).all()
    
    feature_engineer = FeatureEngineer(db)
    feature_data = []
    
    for phone in phones:
        features = await feature_engineer.engineer_comprehensive_features(phone)
        feature_data.append(features)
    
    return pd.DataFrame(feature_data)

# Error handlers
@api_bp.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({'error': 'Rate limit exceeded', 'retry_after': e.retry_after}), 429

@api_bp.errorhandler(400)
def bad_request_handler(e):
    return jsonify({'error': 'Bad request'}), 400

@api_bp.errorhandler(401)
def unauthorized_handler(e):
    return jsonify({'error': 'Unauthorized'}), 401

@api_bp.errorhandler(403)
def forbidden_handler(e):
    return jsonify({'error': 'Forbidden'}), 403

@api_bp.errorhandler(404)
def not_found_handler(e):
    return jsonify({'error': 'Not found'}), 404

@api_bp.errorhandler(500)
def internal_error_handler(e):
    return jsonify({'error': 'Internal server error'}), 500