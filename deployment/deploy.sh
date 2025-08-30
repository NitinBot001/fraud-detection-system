#!/bin/bash

# Fraud Detection System Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
PROJECT_NAME="fraud-detection"
BACKUP_DIR="/backup/fraud-detection"

echo -e "${GREEN}🚀 Starting Fraud Detection System Deployment${NC}"
echo -e "${YELLOW}Environment: $ENVIRONMENT${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check requirements
check_requirements() {
    print_status "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    print_status "Requirements check passed"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create environment file if it doesn't exist
    if [ ! -f ".env.${ENVIRONMENT}" ]; then
        print_warning "Creating default environment file"
        cp .env.example .env.${ENVIRONMENT}
        print_warning "Please update .env.${ENVIRONMENT} with your configuration"
    fi
    
    # Copy environment file
    cp .env.${ENVIRONMENT} .env
    
    # Create necessary directories
    mkdir -p logs data models docker/ssl
    
    # Generate SSL certificates if they don't exist
    if [ ! -f "docker/ssl/cert.pem" ]; then
        print_status "Generating self-signed SSL certificates..."
        openssl req -x509 -newkey rsa:4096 -keyout docker/ssl/key.pem -out docker/ssl/cert.pem \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=fraud-detection.local"
    fi
}

# Database backup
backup_database() {
    if [ "$ENVIRONMENT" = "production" ]; then
        print_status "Creating database backup..."
        
        # Create backup directory
        mkdir -p $BACKUP_DIR
        
        # Backup database
        docker-compose exec -T postgres pg_dump -U fraud_user fraud_detection > \
            "$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"
        
        print_status "Database backup completed"
    fi
}

# Deploy application
deploy_application() {
    print_status "Deploying application..."
    
    # Pull latest images
    docker-compose pull
    
    # Build application
    docker-compose build --no-cache
    
    # Stop existing containers
    docker-compose down --remove-orphans
    
    # Start services
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose up -d
    else
        docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
    fi
    
    print_status "Application deployed successfully"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 10
    
    # Run migrations
    docker-compose exec fraud-detection-app python migrations/create_tables.py
    
    print_status "Database migrations completed"
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    # Wait for application to start
    sleep 30
    
    # Check application health
    if curl -f http://localhost:5000/health &> /dev/null; then
        print_status "Application is healthy"
    else
        print_error "Application health check failed"
        exit 1
    fi
    
    # Check database connection
    if docker-compose exec -T postgres pg_isready -U fraud_user -d fraud_detection &> /dev/null; then
        print_status "Database is healthy"
    else
        print_error "Database health check failed"
        exit 1
    fi
    
    # Check Redis connection
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        print_status "Redis is healthy"
    else
        print_error "Redis health check failed"
        exit 1
    fi
}

# Setup monitoring (optional)
setup_monitoring() {
    if [ "$ENVIRONMENT" = "production" ]; then
        read -p "Do you want to enable monitoring (Prometheus/Grafana)? (y/N): " enable_monitoring
        if [[ $enable_monitoring =~ ^[Yy]$ ]]; then
            print_status "Setting up monitoring..."
            docker-compose --profile monitoring up -d
            print_status "Monitoring setup completed"
            print_status "Grafana available at: http://localhost:3000 (admin/admin)"
            print_status "Prometheus available at: http://localhost:9090"
        fi
    fi
}

# Main deployment flow
main() {
    check_requirements
    setup_environment
    
    if [ "$ENVIRONMENT" = "production" ]; then
        backup_database
    fi
    
    deploy_application
    run_migrations
    health_check
    setup_monitoring
    
    print_status "🎉 Deployment completed successfully!"
    print_status "Application available at: https://localhost"
    print_status "API documentation: https://localhost/api/v1/"
    
    if [ "$ENVIRONMENT" = "development" ]; then
        print_status "Development mode - Application also available at: http://localhost:5000"
    fi
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker-compose down --remove-orphans
    docker system prune -f
}

# Handle script arguments
case "${2:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        cleanup
        ;;
    "backup")
        backup_database
        ;;
    "health")
        health_check
        ;;
    *)
        echo "Usage: $0 [environment] [action]"
        echo "Environment: development|production (default: production)"
        echo "Actions: deploy|cleanup|backup|health (default: deploy)"
        exit 1
        ;;
esac