import asyncio
import uvloop
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
from datetime import timedelta

# Import configurations and components
from config.settings import config
from config.database_config import init_database
from app.api.routes import api_bp
from app.utils.logger import setup_logging, get_logger
from app.services.cache_service import CacheService
from app.services.notification_service import NotificationService
from app.services.data_collector import DataCollector
from app.utils.security import SecurityUtils

# Set up asyncio event loop policy for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

def create_app(config_name='default'):
    """
    Application factory pattern
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    init_extensions(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Initialize database
    init_database()
    
    # Set up logging
    setup_logging()
    
    # Initialize background services
    init_background_services(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app

def init_extensions(app):
    """
    Initialize Flask extensions
    """
    # CORS
    CORS(app, origins=["http://localhost:3000", "https://your-domain.com"])
    
    # JWT
    jwt = JWTManager(app)
    
    # Rate limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        storage_uri=app.config['REDIS_URL'],
        default_limits=["1000 per hour"]
    )
    
    # Additional JWT configuration
    @jwt.additional_claims_loader
    def add_claims_to_access_token(identity):
        from app.models.user_models import User
        from config.database_config import SessionLocal
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(id=identity).first()
            if user:
                return {
                    'username': user.username,
                    'role': user.role,
                    'is_admin': user.is_admin
                }
        finally:
            db.close()
        
        return {}
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return {'error': 'Token has expired'}, 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return {'error': 'Invalid token'}, 401

def register_blueprints(app):
    """
    Register application blueprints
    """
    # API routes
    app.register_blueprint(api_bp)
    
    # Web interface routes (if implemented)
    from app.web import web_bp
    app.register_blueprint(web_bp)

def init_background_services(app):
    """
    Initialize background services
    """
    logger = get_logger(__name__)
    
    # Initialize services
    cache_service = CacheService()
    notification_service = NotificationService()
    
    # Health check for external services
    @app.before_first_request
    def check_external_services():
        # Check cache health
        health = asyncio.run(cache_service.health_check())
        if health['status'] != 'healthy':
            logger.warning(f"Cache service unhealthy: {health}")
        
        # Check other external services
        logger.info("Application startup complete")
    
    # Background tasks
    def start_background_tasks():
        """Start background tasks in separate thread"""
        import threading
        
        def run_data_collection():
            """Run continuous data collection"""
            from config.database_config import SessionLocal
            db = SessionLocal()
            try:
                collector = DataCollector(db)
                asyncio.run(collector.start_continuous_collection())
            finally:
                db.close()
        
        # Start data collection in background thread
        if app.config.get('ENABLE_DATA_COLLECTION', True):
            collection_thread = threading.Thread(target=run_data_collection, daemon=True)
            collection_thread.start()
            logger.info("Background data collection started")
    
    # Start background tasks
    start_background_tasks()

def register_error_handlers(app):
    """
    Register error handlers
    """
    logger = get_logger(__name__)
    
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}")
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {str(error)}", exc_info=True)
        return {'error': 'An unexpected error occurred'}, 500

# Create web blueprint for HTML routes
from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from flask_login import login_required, current_user

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Home page - redirect to dashboard if authenticated"""
    return redirect(url_for('web.dashboard'))

@web_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@web_bp.route('/fraud-analysis')
@login_required
def fraud_analysis():
    """Fraud analysis page"""
    return render_template('fraud_analysis.html')

@web_bp.route('/reports')
@login_required
def reports():
    """Reports page"""
    return render_template('reports.html')

@web_bp.route('/network-analysis')
@login_required
def network_analysis():
    """Network analysis page"""
    return render_template('network_analysis.html')

@web_bp.route('/statistics')
@login_required
def statistics():
    """Statistics page"""
    return render_template('statistics.html')

@web_bp.route('/batch-analysis')
@login_required
def batch_analysis():
    """Batch analysis page"""
    return render_template('batch_analysis.html')

@web_bp.route('/admin')
@login_required
def admin():
    """Admin page"""
    if not current_user.is_admin:
        return redirect(url_for('web.dashboard'))
    return render_template('admin.html')

@web_bp.route('/model-management')
@login_required
def model_management():
    """ML model management page"""
    if not current_user.is_admin:
        return redirect(url_for('web.dashboard'))
    return render_template('model_management.html')

@web_bp.route('/system-health')
@login_required
def system_health():
    """System health monitoring page"""
    if not current_user.is_admin:
        return redirect(url_for('web.dashboard'))
    return render_template('system_health.html')

# WebSocket support for real-time updates
from flask_socketio import SocketIO, emit

def create_socketio_app(app):
    """
    Create SocketIO app for real-time features
    """
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @socketio.on('connect')
    def handle_connect():
        logger = get_logger(__name__)
        logger.info(f"Client connected: {request.sid}")
        emit('status', {'message': 'Connected to fraud detection system'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger = get_logger(__name__)
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('subscribe_alerts')
    def handle_subscribe_alerts():
        """Subscribe to real-time fraud alerts"""
        # Add client to alerts room
        from flask_socketio import join_room
        join_room('fraud_alerts')
        emit('status', {'message': 'Subscribed to fraud alerts'})
    
    return socketio

# Health check endpoint
@web_bp.route('/health')
def health_check():
    """System health check endpoint"""
    import psutil
    from datetime import datetime
    
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application metrics
        cache_service = CacheService()
        cache_health = asyncio.run(cache_service.health_check())
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'uptime': (datetime.now() - psutil.boot_time()).total_seconds()
            },
            'services': {
                'cache': cache_health['status'],
                'database': 'healthy'  # Could add actual DB health check
            }
        }
        
        # Determine overall health
        if (cpu_percent > 90 or memory.percent > 90 or 
            cache_health['status'] != 'healthy'):
            health_data['status'] = 'degraded'
        
        status_code = 200 if health_data['status'] == 'healthy' else 503
        return jsonify(health_data), status_code
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

# Performance monitoring middleware
@web_bp.before_request
def before_request():
    """Track request start time"""
    request.start_time = datetime.now()

@web_bp.after_request
def after_request(response):
    """Log request performance"""
    if hasattr(request, 'start_time'):
        duration = (datetime.now() - request.start_time).total_seconds()
        
        # Log slow requests
        if duration > 1.0:  # Log requests slower than 1 second
            logger = get_logger('performance')
            logger.warning(f"Slow request: {request.method} {request.path} took {duration:.2f}s")
    
    return response

if __name__ == '__main__':
    import os
    
    # Get configuration from environment
    config_name = os.environ.get('FLASK_ENV', 'development')
    
    # Create application
    app = create_app(config_name)
    
    # Create SocketIO app for real-time features
    socketio = create_socketio_app(app)
    
    # Get host and port from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = config_name == 'development'
    
    print(f"""
    🚀 Fraud Detection System Starting...
    
    Environment: {config_name}
    Host: {host}
    Port: {port}
    Debug: {debug}
    
    Dashboard: http://{host}:{port}/dashboard
    API Docs: http://{host}:{port}/api/v1/
    Health: http://{host}:{port}/health
    
    """)
    
    # Run the application
    if config_name == 'development':
        # Development server with auto-reload
        socketio.run(app, host=host, port=port, debug=debug, use_reloader=True)
    else:
        # Production server
        socketio.run(app, host=host, port=port, debug=False)