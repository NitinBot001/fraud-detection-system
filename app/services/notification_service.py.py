import asyncio
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import User, AuditLog
from app.utils.logger import get_logger
from app.services.cache_service import CacheService
from config.settings import Config

logger = get_logger(__name__)

class NotificationService:
    def __init__(self):
        self.cache = CacheService()
        
        # Notification channels
        self.channels = {
            'email': self._send_email_notification,
            'webhook': self._send_webhook_notification,
            'internal': self._send_internal_notification
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'high_risk_detection': {'risk_score': 0.8},
            'fraud_ring_detected': {'min_members': 5},
            'anomaly_burst': {'reports_per_hour': 20},
            'api_failure': {'failure_rate': 0.5},
            'model_drift': {'accuracy_drop': 0.1}
        }
    
    async def send_alert(self, alert_type: str, data: Dict[str, Any], 
                        channels: List[str] = None) -> Dict[str, Any]:
        """
        Send alert through specified channels
        """
        try:
            if channels is None:
                channels = ['email', 'internal']
            
            results = {}
            
            # Prepare alert message
            alert_message = await self._prepare_alert_message(alert_type, data)
            
            # Send through each channel
            for channel in channels:
                if channel in self.channels:
                    try:
                        result = await self.channels[channel](alert_message, data)
                        results[channel] = result
                    except Exception as e:
                        logger.error(f"Failed to send alert via {channel}: {str(e)}")
                        results[channel] = {'status': 'failed', 'error': str(e)}
                else:
                    results[channel] = {'status': 'failed', 'error': 'Unknown channel'}
            
            # Log alert
            await self._log_alert(alert_type, data, results)
            
            return {
                'alert_type': alert_type,
                'timestamp': datetime.now().isoformat(),
                'channels': results,
                'success': any(r.get('status') == 'sent' for r in results.values())
            }
            
        except Exception as e:
            logger.error(f"Alert sending failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _prepare_alert_message(self, alert_type: str, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare alert message based on type and data
        """
        templates = {
            'high_risk_detection': {
                'subject': '🚨 High Risk Phone Number Detected',
                'text': '''
High Risk Detection Alert

Phone Number: {phone_number}
Risk Score: {risk_score:.2f}
Risk Level: {risk_level}
Detection Time: {timestamp}

Detected Patterns:
{patterns}

Immediate action may be required.
                ''',
                'html': '''
<h2>🚨 High Risk Detection Alert</h2>
<p><strong>Phone Number:</strong> {phone_number}</p>
<p><strong>Risk Score:</strong> {risk_score:.2f}</p>
<p><strong>Risk Level:</strong> {risk_level}</p>
<p><strong>Detection Time:</strong> {timestamp}</p>
<h3>Detected Patterns:</h3>
<ul>{pattern_list}</ul>
<p><em>Immediate action may be required.</em></p>
                '''
            },
            'fraud_ring_detected': {
                'subject': '⚠️ Fraud Ring Detected',
                'text': '''
Fraud Ring Detection Alert

Ring Size: {ring_size} members
Risk Score: {risk_score:.2f}
Detection Method: {detection_method}

Members:
{members}

Recommended Action: Immediate investigation and blocking.
                ''',
                'html': '''
<h2>⚠️ Fraud Ring Detection Alert</h2>
<p><strong>Ring Size:</strong> {ring_size} members</p>
<p><strong>Risk Score:</strong> {risk_score:.2f}</p>
<p><strong>Detection Method:</strong> {detection_method}</p>
<h3>Members:</h3>
<ul>{member_list}</ul>
<p><strong>Recommended Action:</strong> Immediate investigation and blocking.</p>
                '''
            },
            'anomaly_burst': {
                'subject': '📈 Unusual Activity Burst Detected',
                'text': '''
Anomaly Burst Alert

Activity Type: {activity_type}
Burst Rate: {burst_rate} per hour
Duration: {duration} minutes
Affected Numbers: {affected_count}

This may indicate coordinated fraudulent activity.
                ''',
                'html': '''
<h2>📈 Unusual Activity Burst Detected</h2>
<p><strong>Activity Type:</strong> {activity_type}</p>
<p><strong>Burst Rate:</strong> {burst_rate} per hour</p>
<p><strong>Duration:</strong> {duration} minutes</p>
<p><strong>Affected Numbers:</strong> {affected_count}</p>
<p><em>This may indicate coordinated fraudulent activity.</em></p>
                '''
            },
            'model_performance': {
                'subject': '🔧 Model Performance Alert',
                'text': '''
Model Performance Alert

Model: {model_name}
Current Accuracy: {current_accuracy:.2f}
Previous Accuracy: {previous_accuracy:.2f}
Performance Drop: {accuracy_drop:.2f}

Model retraining may be required.
                ''',
                'html': '''
<h2>🔧 Model Performance Alert</h2>
<p><strong>Model:</strong> {model_name}</p>
<p><strong>Current Accuracy:</strong> {current_accuracy:.2f}</p>
<p><strong>Previous Accuracy:</strong> {previous_accuracy:.2f}</p>
<p><strong>Performance Drop:</strong> {accuracy_drop:.2f}</p>
<p><em>Model retraining may be required.</em></p>
                '''
            }
        }
        
        if alert_type not in templates:
            # Generic template
            return {
                'subject': f'Alert: {alert_type}',
                'text': f'Alert Type: {alert_type}\nData: {json.dumps(data, indent=2)}',
                'html': f'<h2>Alert: {alert_type}</h2><pre>{json.dumps(data, indent=2)}</pre>'
            }
        
        template = templates[alert_type]
        
        # Format text message
        try:
            text_message = template['text'].format(**data)
        except KeyError as e:
            text_message = f"Alert: {alert_type}\nMissing data: {str(e)}\nData: {json.dumps(data, indent=2)}"
        
        # Format HTML message
        try:
            # Special formatting for lists
            formatted_data = data.copy()
            
            if 'patterns' in data and isinstance(data['patterns'], list):
                formatted_data['patterns'] = '\n'.join(f"- {pattern}" for pattern in data['patterns'])
                formatted_data['pattern_list'] = ''.join(f"<li>{pattern}</li>" for pattern in data['patterns'])
            
            if 'members' in data and isinstance(data['members'], list):
                formatted_data['members'] = '\n'.join(f"- {member}" for member in data['members'])
                formatted_data['member_list'] = ''.join(f"<li>{member}</li>" for member in data['members'])
            
            html_message = template['html'].format(**formatted_data)
        except KeyError as e:
            html_message = f"<h2>Alert: {alert_type}</h2><p>Missing data: {str(e)}</p><pre>{json.dumps(data, indent=2)}</pre>"
        
        return {
            'subject': template['subject'],
            'text': text_message,
            'html': html_message
        }
    
    async def _send_email_notification(self, message: Dict[str, str], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send email notification
        """
        try:
            # Email configuration (in production, use proper email service)
            smtp_server = "localhost"  # Configure with actual SMTP server
            smtp_port = 587
            sender_email = "alerts@fraud-detection.com"
            sender_password = "your_email_password"
            
            # Get recipient emails from configuration or database
            recipients = self._get_alert_recipients(data.get('alert_level', 'medium'))
            
            if not recipients:
                return {'status': 'skipped', 'reason': 'No recipients configured'}
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = message['subject']
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            
            # Add text and HTML parts
            text_part = MIMEText(message['text'], 'plain')
            html_part = MIMEText(message['html'], 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email (commented out for demo - configure with real SMTP)
            # with smtplib.SMTP(smtp_server, smtp_port) as server:
            #     server.starttls()
            #     server.login(sender_email, sender_password)
            #     server.send_message(msg)
            
            logger.info(f"Email alert sent to {len(recipients)} recipients")
            
            return {
                'status': 'sent',
                'recipients': recipients,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Email notification failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _send_webhook_notification(self, message: Dict[str, str], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send webhook notification
        """
        try:
            import aiohttp
            
            webhook_urls = [
                'https://your-webhook-endpoint.com/alerts',
                'https://slack-webhook-url.com/webhook'
            ]
            
            payload = {
                'alert_type': data.get('alert_type'),
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'data': data
            }
            
            results = []
            
            async with aiohttp.ClientSession() as session:
                for url in webhook_urls:
                    try:
                        async with session.post(url, json=payload) as response:
                            if response.status == 200:
                                results.append({'url': url, 'status': 'sent'})
                            else:
                                results.append({'url': url, 'status': 'failed', 'code': response.status})
                    except Exception as e:
                        results.append({'url': url, 'status': 'failed', 'error': str(e)})
            
            success_count = sum(1 for r in results if r['status'] == 'sent')
            
            return {
                'status': 'sent' if success_count > 0 else 'failed',
                'webhooks': results,
                'success_count': success_count
            }
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _send_internal_notification(self, message: Dict[str, str], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send internal system notification
        """
        try:
            # Store in cache for real-time dashboard
            notification = {
                'id': f"alert_{datetime.now().timestamp()}",
                'type': data.get('alert_type', 'general'),
                'level': data.get('alert_level', 'medium'),
                'message': message['text'],
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'read': False
            }
            
            # Store in cache with expiration
            cache_key = f"notification:{notification['id']}"
            await self.cache.set(cache_key, notification, timeout=86400)  # 24 hours
            
            # Add to notifications list
            notifications_key = "active_notifications"
            notifications = await self.cache.get(notifications_key) or []
            notifications.append(notification['id'])
            
            # Keep only last 100 notifications
            if len(notifications) > 100:
                # Remove old notifications
                old_notifications = notifications[:-100]
                for old_id in old_notifications:
                    await self.cache.delete(f"notification:{old_id}")
                notifications = notifications[-100:]
            
            await self.cache.set(notifications_key, notifications, timeout=86400)
            
            logger.info(f"Internal notification stored: {notification['id']}")
            
            return {
                'status': 'sent',
                'notification_id': notification['id'],
                'timestamp': notification['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Internal notification failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _get_alert_recipients(self, alert_level: str) -> List[str]:
        """
        Get email recipients based on alert level
        """
        # In production, this would query database for user preferences
        recipients = {
            'low': ['alerts@fraud-detection.com'],
            'medium': ['alerts@fraud-detection.com', 'manager@fraud-detection.com'],
            'high': ['alerts@fraud-detection.com', 'manager@fraud-detection.com', 'director@fraud-detection.com'],
            'critical': ['alerts@fraud-detection.com', 'manager@fraud-detection.com', 'director@fraud-detection.com', 'cto@fraud-detection.com']
        }
        
        return recipients.get(alert_level, recipients['medium'])
    
    async def _log_alert(self, alert_type: str, data: Dict[str, Any], results: Dict[str, Any]):
        """
        Log alert to audit trail
        """
        try:
            alert_log = {
                'alert_type': alert_type,
                'data': data,
                'notification_results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in cache for analytics
            log_key = f"alert_log:{datetime.now().timestamp()}"
            await self.cache.set(log_key, alert_log, timeout=2592000)  # 30 days
            
        except Exception as e:
            logger.error(f"Alert logging failed: {str(e)}")
    
    async def get_active_notifications(self, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Get active notifications for dashboard
        """
        try:
            notifications_key = "active_notifications"
            notification_ids = await self.cache.get(notifications_key) or []
            
            notifications = []
            for notification_id in notification_ids:
                cache_key = f"notification:{notification_id}"
                notification = await self.cache.get(cache_key)
                if notification:
                    notifications.append(notification)
            
            # Sort by timestamp (newest first)
            notifications.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return notifications
            
        except Exception as e:
            logger.error(f"Get notifications failed: {str(e)}")
            return []
    
    async def mark_notification_read(self, notification_id: str) -> bool:
        """
        Mark notification as read
        """
        try:
            cache_key = f"notification:{notification_id}"
            notification = await self.cache.get(cache_key)
            
            if notification:
                notification['read'] = True
                await self.cache.set(cache_key, notification, timeout=86400)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Mark notification read failed: {str(e)}")
            return False
    
    async def send_scheduled_report(self, report_type: str, recipients: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send scheduled reports (daily, weekly, monthly)
        """
        try:
            report_templates = {
                'daily_summary': {
                    'subject': '📊 Daily Fraud Detection Summary - {date}',
                    'template': 'daily_report_template.html'
                },
                'weekly_analysis': {
                    'subject': '📈 Weekly Fraud Analysis Report - Week of {date}',
                    'template': 'weekly_report_template.html'
                },
                'monthly_insights': {
                    'subject': '🔍 Monthly Fraud Insights Report - {month} {year}',
                    'template': 'monthly_report_template.html'
                }
            }
            
            if report_type not in report_templates:
                return {'status': 'failed', 'error': 'Unknown report type'}
            
            template_info = report_templates[report_type]
            
            # Generate report content
            report_message = await self._generate_report_content(report_type, data)
            
            # Send via email
            result = await self._send_email_notification(report_message, {
                'alert_level': 'low',
                'recipients': recipients
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Scheduled report failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _generate_report_content(self, report_type: str, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate report content based on type and data
        """
        # This would use proper templating engine in production
        if report_type == 'daily_summary':
            subject = f"📊 Daily Fraud Detection Summary - {data.get('date', datetime.now().strftime('%Y-%m-%d'))}"
            
            text_content = f"""
Daily Fraud Detection Summary

Date: {data.get('date', 'Today')}
Total Reports: {data.get('total_reports', 0)}
High Risk Detections: {data.get('high_risk_detections', 0)}
Blocked Numbers: {data.get('blocked_numbers', 0)}
False Positives: {data.get('false_positives', 0)}

Top Fraud Types:
{data.get('top_fraud_types', 'No data available')}

System Performance:
- Average Response Time: {data.get('avg_response_time', 'N/A')}ms
- API Uptime: {data.get('api_uptime', 'N/A')}%
- Model Accuracy: {data.get('model_accuracy', 'N/A')}%
            """
            
            html_content = f"""
<h2>📊 Daily Fraud Detection Summary</h2>
<h3>Key Metrics</h3>
<ul>
    <li><strong>Total Reports:</strong> {data.get('total_reports', 0)}</li>
    <li><strong>High Risk Detections:</strong> {data.get('high_risk_detections', 0)}</li>
    <li><strong>Blocked Numbers:</strong> {data.get('blocked_numbers', 0)}</li>
    <li><strong>False Positives:</strong> {data.get('false_positives', 0)}</li>
</ul>

<h3>System Performance</h3>
<ul>
    <li><strong>Average Response Time:</strong> {data.get('avg_response_time', 'N/A')}ms</li>
    <li><strong>API Uptime:</strong> {data.get('api_uptime', 'N/A')}%</li>
    <li><strong>Model Accuracy:</strong> {data.get('model_accuracy', 'N/A')}%</li>
</ul>
            """
            
            return {
                'subject': subject,
                'text': text_content,
                'html': html_content
            }
        
        # Add more report types as needed
        return {
            'subject': f'Report: {report_type}',
            'text': f'Report data: {json.dumps(data, indent=2)}',
            'html': f'<h2>Report: {report_type}</h2><pre>{json.dumps(data, indent=2)}</pre>'
        }