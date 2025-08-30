import networkx as nx
from typing import Dict, List, Tuple, Set
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import PhoneNumber, NetworkConnection, FraudReport
from app.utils.logger import get_logger
import asyncio

logger = get_logger(__name__)

class NetworkAnalyzer:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.fraud_propagation_threshold = 0.7
        self.network_cache = {}
    
    async def analyze_phone_network(self, phone_record: PhoneNumber) -> Dict:
        """
        Comprehensive network analysis for a phone number
        """
        try:
            # Build network graph
            network = await self._build_network_graph(phone_record)
            
            # Calculate network metrics
            metrics = await self._calculate_network_metrics(network, phone_record.number)
            
            # Detect suspicious patterns
            patterns = await self._detect_network_patterns(network, phone_record.number)
            
            # Calculate fraud propagation risk
            propagation_risk = await self._calculate_fraud_propagation(network, phone_record.number)
            
            # Identify key network nodes
            key_nodes = await self._identify_key_nodes(network)
            
            return {
                'network_risk': metrics.get('network_risk', 0.0),
                'network_size': metrics.get('network_size', 0),
                'clustering_coefficient': metrics.get('clustering_coefficient', 0.0),
                'centrality_score': metrics.get('centrality_score', 0.0),
                'suspicious_patterns': patterns,
                'fraud_propagation_risk': propagation_risk,
                'key_connected_nodes': key_nodes,
                'network_metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Network analysis failed for {phone_record.number}: {str(e)}")
            return {'network_risk': 0.0, 'suspicious_patterns': []}
    
    async def _build_network_graph(self, phone_record: PhoneNumber, depth: int = 2) -> nx.Graph:
        """
        Build network graph around the phone number
        """
        graph = nx.Graph()
        visited = set()
        queue = [(phone_record.number, 0)]
        
        while queue:
            current_phone, current_depth = queue.pop(0)
            
            if current_phone in visited or current_depth > depth:
                continue
                
            visited.add(current_phone)
            graph.add_node(current_phone)
            
            # Get connections for current phone
            connections = await self._get_phone_connections(current_phone)
            
            for connection in connections:
                target_phone = connection.get('target_phone')
                if target_phone and target_phone not in visited:
                    graph.add_edge(
                        current_phone, 
                        target_phone,
                        weight=connection.get('strength', 1.0),
                        connection_type=connection.get('connection_type'),
                        frequency=connection.get('frequency', 1)
                    )
                    
                    if current_depth < depth:
                        queue.append((target_phone, current_depth + 1))
        
        return graph
    
    async def _get_phone_connections(self, phone_number: str) -> List[Dict]:
        """
        Get all connections for a phone number
        """
        # Get phone record
        phone_record = self.db.query(PhoneNumber).filter_by(number=phone_number).first()
        if not phone_record:
            return []
        
        # Query direct connections
        connections = self.db.query(NetworkConnection).filter(
            (NetworkConnection.source_phone_id == phone_record.id) |
            (NetworkConnection.target_phone_id == phone_record.id)
        ).all()
        
        result = []
        for conn in connections:
            # Determine target phone
            if conn.source_phone_id == phone_record.id:
                target_record = self.db.query(PhoneNumber).filter_by(id=conn.target_phone_id).first()
            else:
                target_record = self.db.query(PhoneNumber).filter_by(id=conn.source_phone_id).first()
            
            if target_record:
                result.append({
                    'target_phone': target_record.number,
                    'connection_type': conn.connection_type,
                    'strength': conn.strength,
                    'frequency': conn.frequency,
                    'last_interaction': conn.last_interaction
                })
        
        return result
    
    async def _calculate_network_metrics(self, graph: nx.Graph, phone_number: str) -> Dict:
        """
        Calculate various network metrics
        """
        if not graph.has_node(phone_number):
            return {'network_risk': 0.0, 'network_size': 0}
        
        metrics = {}
        
        # Basic metrics
        metrics['network_size'] = len(graph.nodes())
        metrics['edge_count'] = len(graph.edges())
        
        # Centrality measures
        try:
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            closeness_centrality = nx.closeness_centrality(graph)
            
            metrics['degree_centrality'] = degree_centrality.get(phone_number, 0.0)
            metrics['betweenness_centrality'] = betweenness_centrality.get(phone_number, 0.0)
            metrics['closeness_centrality'] = closeness_centrality.get(phone_number, 0.0)
            
            # Combined centrality score
            metrics['centrality_score'] = (
                metrics['degree_centrality'] * 0.4 +
                metrics['betweenness_centrality'] * 0.3 +
                metrics['closeness_centrality'] * 0.3
            )
        except:
            metrics['centrality_score'] = 0.0
        
        # Clustering coefficient
        try:
            clustering = nx.clustering(graph)
            metrics['clustering_coefficient'] = clustering.get(phone_number, 0.0)
        except:
            metrics['clustering_coefficient'] = 0.0
        
        # Calculate network risk based on various factors
        risk_factors = []
        
        # High centrality in a dense network is suspicious
        if metrics['centrality_score'] > 0.8 and metrics['network_size'] > 50:
            risk_factors.append(0.6)
        
        # Many connections with high clustering (potential fraud ring)
        if metrics['degree_centrality'] > 0.7 and metrics['clustering_coefficient'] > 0.8:
            risk_factors.append(0.7)
        
        # Large network size
        if metrics['network_size'] > 100:
            risk_factors.append(0.4)
        
        metrics['network_risk'] = min(1.0, max(risk_factors) if risk_factors else 0.0)
        
        return metrics
    
    async def _detect_network_patterns(self, graph: nx.Graph, phone_number: str) -> List[str]:
        """
        Detect suspicious network patterns
        """
        patterns = []
        
        if not graph.has_node(phone_number):
            return patterns
        
        # Detect cliques (potential fraud rings)
        cliques = list(nx.find_cliques(graph))
        large_cliques = [c for c in cliques if len(c) >= 5 and phone_number in c]
        if large_cliques:
            patterns.append(f"Member of {len(large_cliques)} potential fraud rings")
        
        # Detect star patterns (one node connected to many)
        neighbors = list(graph.neighbors(phone_number))
        if len(neighbors) > 20:
            patterns.append("Hub pattern detected - connected to many numbers")
        
        # Detect bridge nodes (connecting different communities)
        try:
            bridges = list(nx.bridges(graph))
            for bridge in bridges:
                if phone_number in bridge:
                    patterns.append("Bridge node - connects separate groups")
                    break
        except:
            pass
        
        # Detect rapid network expansion
        # This would require temporal analysis of network growth
        
        return patterns
    
    async def _calculate_fraud_propagation(self, graph: nx.Graph, phone_number: str) -> float:
        """
        Calculate fraud propagation risk
        """
        if not graph.has_node(phone_number):
            return 0.0
        
        # Get fraud scores of connected nodes
        connected_phones = list(graph.neighbors(phone_number))
        fraud_scores = []
        
        for connected_phone in connected_phones:
            # Get fraud reports for connected phone
            phone_record = self.db.query(PhoneNumber).filter_by(number=connected_phone).first()
            if phone_record:
                report_count = self.db.query(FraudReport).filter_by(
                    phone_number_id=phone_record.id,
                    status='VERIFIED'
                ).count()
                
                # Simple fraud score based on verified reports
                fraud_score = min(1.0, report_count / 5.0)
                fraud_scores.append(fraud_score)
        
        if not fraud_scores:
            return 0.0
        
        # Calculate propagation risk
        avg_fraud_score = sum(fraud_scores) / len(fraud_scores)
        max_fraud_score = max(fraud_scores)
        
        # Higher risk if connected to many fraudulent numbers
        propagation_risk = (avg_fraud_score * 0.6 + max_fraud_score * 0.4)
        
        # Amplify if many connections are fraudulent
        fraud_ratio = sum(1 for score in fraud_scores if score > 0.5) / len(fraud_scores)
        if fraud_ratio > 0.3:
            propagation_risk *= 1.5
        
        return min(1.0, propagation_risk)
    
    async def _identify_key_nodes(self, graph: nx.Graph) -> List[Dict]:
        """
        Identify key nodes in the network
        """
        if len(graph.nodes()) == 0:
            return []
        
        key_nodes = []
        
        try:
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            
            # Sort by combined centrality score
            combined_scores = {}
            for node in graph.nodes():
                combined_scores[node] = (
                    degree_centrality.get(node, 0) * 0.6 +
                    betweenness_centrality.get(node, 0) * 0.4
                )
            
            # Get top 10 key nodes
            sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for node, score in sorted_nodes:
                key_nodes.append({
                    'phone_number': node,
                    'centrality_score': score,
                    'degree': graph.degree(node),
                    'is_suspected_fraud': score > 0.8  # Threshold for suspicion
                })
                
        except Exception as e:
            logger.error(f"Error identifying key nodes: {str(e)}")
        
        return key_nodes
    
    async def detect_fraud_rings(self, min_size: int = 5) -> List[Dict]:
        """
        Detect potential fraud rings in the network
        """
        # Build comprehensive network
        all_phones = self.db.query(PhoneNumber).all()
        full_graph = nx.Graph()
        
        # Add all connections to graph
        connections = self.db.query(NetworkConnection).all()
        for conn in connections:
            source_phone = self.db.query(PhoneNumber).filter_by(id=conn.source_phone_id).first()
            target_phone = self.db.query(PhoneNumber).filter_by(id=conn.target_phone_id).first()
            
            if source_phone and target_phone:
                full_graph.add_edge(
                    source_phone.number,
                    target_phone.number,
                    weight=conn.strength
                )
        
        # Detect communities/cliques
        try:
            # Use community detection algorithms
            communities = nx.community.greedy_modularity_communities(full_graph)
            
            fraud_rings = []
            for i, community in enumerate(communities):
                if len(community) >= min_size:
                    # Check if community has high fraud rate
                    fraud_count = 0
                    for phone in community:
                        phone_record = self.db.query(PhoneNumber).filter_by(number=phone).first()
                        if phone_record:
                            report_count = self.db.query(FraudReport).filter_by(
                                phone_number_id=phone_record.id,
                                status='VERIFIED'
                            ).count()
                            if report_count > 0:
                                fraud_count += 1
                    
                    fraud_rate = fraud_count / len(community)
                    if fraud_rate > 0.3:  # 30% of nodes have fraud reports
                        fraud_rings.append({
                            'ring_id': f"ring_{i}",
                            'members': list(community),
                            'size': len(community),
                            'fraud_rate': fraud_rate,
                            'risk_score': min(1.0, fraud_rate * 2)
                        })
            
            return fraud_rings
            
        except Exception as e:
            logger.error(f"Fraud ring detection failed: {str(e)}")
            return []