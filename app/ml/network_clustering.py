import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, modularity
import community as community_louvain
from scipy.sparse import csr_matrix
from collections import defaultdict
import asyncio
from app.utils.logger import get_logger
from app.services.cache_service import CacheService

logger = get_logger(__name__)

class NetworkClustering:
    def __init__(self):
        self.cache = CacheService()
        self.clustering_algorithms = {
            'spectral': SpectralClustering,
            'dbscan': DBSCAN,
            'kmeans': KMeans
        }
        
        # Algorithm parameters
        self.algorithm_params = {
            'spectral': {
                'n_clusters': 10,
                'affinity': 'rbf',
                'random_state': 42
            },
            'dbscan': {
                'eps': 0.3,
                'min_samples': 5
            },
            'kmeans': {
                'n_clusters': 8,
                'random_state': 42,
                'n_init': 10
            }
        }
    
    async def detect_fraud_communities(self, graph: nx.Graph, 
                                     fraud_labels: Dict[str, bool] = None) -> Dict:
        """
        Detect communities in the network and identify fraud clusters
        """
        try:
            if len(graph.nodes()) < 10:
                return {'communities': [], 'fraud_communities': [], 'metrics': {}}
            
            # Community detection using multiple algorithms
            communities = {}
            
            # Louvain algorithm for modularity optimization
            communities['louvain'] = await self._louvain_clustering(graph)
            
            # Spectral clustering
            communities['spectral'] = await self._spectral_clustering(graph)
            
            # Edge betweenness clustering
            communities['edge_betweenness'] = await self._edge_betweenness_clustering(graph)
            
            # Analyze communities for fraud patterns
            fraud_analysis = await self._analyze_communities_for_fraud(
                graph, communities, fraud_labels
            )
            
            # Calculate clustering metrics
            metrics = await self._calculate_clustering_metrics(graph, communities)
            
            return {
                'communities': communities,
                'fraud_analysis': fraud_analysis,
                'metrics': metrics,
                'recommendations': await self._generate_clustering_recommendations(fraud_analysis)
            }
            
        except Exception as e:
            logger.error(f"Community detection failed: {str(e)}")
            return {'communities': [], 'fraud_communities': [], 'metrics': {}}
    
    async def _louvain_clustering(self, graph: nx.Graph) -> Dict:
        """
        Louvain algorithm for community detection
        """
        try:
            # Convert to undirected for community detection
            if graph.is_directed():
                undirected_graph = graph.to_undirected()
            else:
                undirected_graph = graph
            
            # Apply Louvain algorithm
            partition = community_louvain.best_partition(undirected_graph)
            
            # Organize communities
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            # Calculate modularity
            modularity_score = community_louvain.modularity(partition, undirected_graph)
            
            return {
                'algorithm': 'louvain',
                'communities': dict(communities),
                'n_communities': len(communities),
                'modularity': modularity_score,
                'partition': partition
            }
            
        except Exception as e:
            logger.error(f"Louvain clustering failed: {str(e)}")
            return {'algorithm': 'louvain', 'communities': {}, 'n_communities': 0}
    
    async def _spectral_clustering(self, graph: nx.Graph) -> Dict:
        """
        Spectral clustering on graph
        """
        try:
            # Convert graph to adjacency matrix
            nodes = list(graph.nodes())
            adj_matrix = nx.adjacency_matrix(graph, nodelist=nodes)
            
            # Apply spectral clustering
            n_clusters = min(self.algorithm_params['spectral']['n_clusters'], len(nodes) // 2)
            
            if n_clusters < 2:
                return {'algorithm': 'spectral', 'communities': {}, 'n_communities': 0}
            
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            cluster_labels = spectral.fit_predict(adj_matrix.toarray())
            
            # Organize communities
            communities = defaultdict(list)
            for node, label in zip(nodes, cluster_labels):
                communities[label].append(node)
            
            return {
                'algorithm': 'spectral',
                'communities': dict(communities),
                'n_communities': len(communities),
                'labels': dict(zip(nodes, cluster_labels))
            }
            
        except Exception as e:
            logger.error(f"Spectral clustering failed: {str(e)}")
            return {'algorithm': 'spectral', 'communities': {}, 'n_communities': 0}
    
    async def _edge_betweenness_clustering(self, graph: nx.Graph) -> Dict:
        """
        Community detection using edge betweenness
        """
        try:
            # Create a copy to avoid modifying original
            g = graph.copy()
            
            # Iteratively remove edges with highest betweenness
            communities = []
            max_iterations = min(50, len(g.edges()) // 2)
            
            for _ in range(max_iterations):
                if len(g.edges()) == 0:
                    break
                
                # Calculate edge betweenness
                edge_betweenness = nx.edge_betweenness_centrality(g)
                
                if not edge_betweenness:
                    break
                
                # Remove edge with highest betweenness
                max_edge = max(edge_betweenness.items(), key=lambda x: x[1])[0]
                g.remove_edge(*max_edge)
                
                # Check for disconnected components
                components = list(nx.connected_components(g))
                if len(components) > 1:
                    communities = {i: list(comp) for i, comp in enumerate(components)}
                    break
            
            return {
                'algorithm': 'edge_betweenness',
                'communities': communities,
                'n_communities': len(communities)
            }
            
        except Exception as e:
            logger.error(f"Edge betweenness clustering failed: {str(e)}")
            return {'algorithm': 'edge_betweenness', 'communities': {}, 'n_communities': 0}
    
    async def _analyze_communities_for_fraud(self, graph: nx.Graph, 
                                           communities: Dict, 
                                           fraud_labels: Dict[str, bool] = None) -> Dict:
        """
        Analyze communities for fraud patterns
        """
        try:
            fraud_analysis = {}
            
            if fraud_labels is None:
                fraud_labels = {}
            
            for algorithm, community_data in communities.items():
                if 'communities' not in community_data:
                    continue
                
                algorithm_analysis = {
                    'fraud_communities': [],
                    'suspicious_communities': [],
                    'clean_communities': []
                }
                
                for community_id, members in community_data['communities'].items():
                    if len(members) < 3:  # Skip very small communities
                        continue
                    
                    # Calculate fraud metrics for this community
                    community_metrics = await self._calculate_community_fraud_metrics(
                        graph, members, fraud_labels
                    )
                    
                    community_info = {
                        'community_id': community_id,
                        'members': members,
                        'size': len(members),
                        'metrics': community_metrics
                    }
                    
                    # Classify community based on fraud metrics
                    if community_metrics['fraud_ratio'] > 0.5:
                        algorithm_analysis['fraud_communities'].append(community_info)
                    elif community_metrics['fraud_ratio'] > 0.2 or community_metrics['risk_score'] > 0.6:
                        algorithm_analysis['suspicious_communities'].append(community_info)
                    else:
                        algorithm_analysis['clean_communities'].append(community_info)
                
                fraud_analysis[algorithm] = algorithm_analysis
            
            return fraud_analysis
            
        except Exception as e:
            logger.error(f"Community fraud analysis failed: {str(e)}")
            return {}
    
    async def _calculate_community_fraud_metrics(self, graph: nx.Graph, 
                                               members: List[str], 
                                               fraud_labels: Dict[str, bool]) -> Dict:
        """
        Calculate fraud-related metrics for a community
        """
        try:
            metrics = {
                'fraud_count': 0,
                'fraud_ratio': 0.0,
                'density': 0.0,
                'centralization': 0.0,
                'risk_score': 0.0
            }
            
            # Count known fraudulent members
            fraud_count = sum(1 for member in members if fraud_labels.get(member, False))
            metrics['fraud_count'] = fraud_count
            metrics['fraud_ratio'] = fraud_count / len(members) if members else 0
            
            # Calculate community density
            subgraph = graph.subgraph(members)
            if len(members) > 1:
                possible_edges = len(members) * (len(members) - 1) / 2
                actual_edges = len(subgraph.edges())
                metrics['density'] = actual_edges / possible_edges if possible_edges > 0 else 0
            
            # Calculate centralization (how centralized is the community)
            if len(members) > 2:
                degrees = [subgraph.degree(node) for node in members]
                max_degree = max(degrees) if degrees else 0
                sum_diff = sum(max_degree - degree for degree in degrees)
                max_possible_sum = (len(members) - 1) * (len(members) - 2)
                metrics['centralization'] = sum_diff / max_possible_sum if max_possible_sum > 0 else 0
            
            # Calculate risk score based on multiple factors
            risk_factors = [
                metrics['fraud_ratio'] * 0.4,  # Known fraud members
                min(1.0, metrics['density'] * 2) * 0.3,  # High density suspicious
                metrics['centralization'] * 0.2,  # High centralization suspicious
                min(1.0, len(members) / 20) * 0.1  # Large communities more suspicious
            ]
            
            metrics['risk_score'] = sum(risk_factors)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Community metrics calculation failed: {str(e)}")
            return {'fraud_count': 0, 'fraud_ratio': 0.0, 'risk_score': 0.0}
    
    async def _calculate_clustering_metrics(self, graph: nx.Graph, communities: Dict) -> Dict:
        """
        Calculate overall clustering quality metrics
        """
        try:
            metrics = {}
            
            for algorithm, community_data in communities.items():
                if 'communities' not in community_data:
                    continue
                
                algorithm_metrics = {
                    'n_communities': community_data.get('n_communities', 0),
                    'modularity': community_data.get('modularity', 0.0)
                }
                
                # Calculate average community size
                community_sizes = [len(members) for members in community_data['communities'].values()]
                if community_sizes:
                    algorithm_metrics['avg_community_size'] = np.mean(community_sizes)
                    algorithm_metrics['max_community_size'] = max(community_sizes)
                    algorithm_metrics['min_community_size'] = min(community_sizes)
                    algorithm_metrics['community_size_variance'] = np.var(community_sizes)
                
                # Calculate coverage (fraction of nodes in communities)
                total_nodes_in_communities = sum(community_sizes)
                algorithm_metrics['coverage'] = total_nodes_in_communities / len(graph.nodes()) if graph.nodes() else 0
                
                metrics[algorithm] = algorithm_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Clustering metrics calculation failed: {str(e)}")
            return {}
    
    async def _generate_clustering_recommendations(self, fraud_analysis: Dict) -> List[str]:
        """
        Generate recommendations based on community analysis
        """
        recommendations = []
        
        try:
            for algorithm, analysis in fraud_analysis.items():
                fraud_communities = analysis.get('fraud_communities', [])
                suspicious_communities = analysis.get('suspicious_communities', [])
                
                if fraud_communities:
                    recommendations.append(
                        f"{algorithm}: {len(fraud_communities)} high-risk communities detected - immediate investigation recommended"
                    )
                
                if suspicious_communities:
                    recommendations.append(
                        f"{algorithm}: {len(suspicious_communities)} suspicious communities - enhanced monitoring recommended"
                    )
                
                # Check for large fraud communities
                large_fraud_communities = [c for c in fraud_communities if c['size'] > 10]
                if large_fraud_communities:
                    recommendations.append(
                        f"{algorithm}: Large fraud ring detected with {max(c['size'] for c in large_fraud_communities)} members"
                    )
            
            # General recommendations
            if not recommendations:
                recommendations.append("No significant fraud communities detected - continue routine monitoring")
            else:
                recommendations.append("Consider blocking or restricting identified fraud communities")
                recommendations.append("Investigate connections between fraud communities")
        
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            recommendations.append("Error in analysis - manual review recommended")
        
        return recommendations
    
    async def cluster_phone_features(self, phone_features: pd.DataFrame) -> Dict:
        """
        Cluster phones based on behavioral features
        """
        try:
            if len(phone_features) < 10:
                return {'clusters': {}, 'metrics': {}}
            
            # Prepare features for clustering
            feature_columns = [col for col in phone_features.columns if col != 'phone_number']
            X = phone_features[feature_columns].fillna(0).values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply multiple clustering algorithms
            clustering_results = {}
            
            # K-means clustering
            kmeans_result = await self._apply_kmeans_clustering(X_scaled, phone_features['phone_number'].values)
            clustering_results['kmeans'] = kmeans_result
            
            # DBSCAN clustering
            dbscan_result = await self._apply_dbscan_clustering(X_scaled, phone_features['phone_number'].values)
            clustering_results['dbscan'] = dbscan_result
            
            # Calculate clustering metrics
            metrics = await self._calculate_feature_clustering_metrics(X_scaled, clustering_results)
            
            return {
                'clusters': clustering_results,
                'metrics': metrics,
                'feature_importance': await self._analyze_feature_importance(X_scaled, feature_columns, clustering_results)
            }
            
        except Exception as e:
            logger.error(f"Feature clustering failed: {str(e)}")
            return {'clusters': {}, 'metrics': {}}
    
    async def _apply_kmeans_clustering(self, X: np.ndarray, phone_numbers: np.ndarray) -> Dict:
        """
        Apply K-means clustering
        """
        try:
            # Determine optimal number of clusters using elbow method
            max_clusters = min(10, len(X) // 5)
            if max_clusters < 2:
                return {'algorithm': 'kmeans', 'clusters': {}}
            
            inertias = []
            silhouette_scores = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                inertias.append(kmeans.inertia_)
                if len(set(labels)) > 1:
                    silhouette_scores.append(silhouette_score(X, labels))
                else:
                    silhouette_scores.append(0)
            
            # Choose optimal k (highest silhouette score)
            optimal_k = np.argmax(silhouette_scores) + 2
            
            # Final clustering with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Organize clusters
            clusters = defaultdict(list)
            for phone, label in zip(phone_numbers, labels):
                clusters[label].append(phone)
            
            return {
                'algorithm': 'kmeans',
                'clusters': dict(clusters),
                'n_clusters': optimal_k,
                'silhouette_score': silhouette_scores[optimal_k - 2],
                'cluster_centers': kmeans.cluster_centers_
            }
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {str(e)}")
            return {'algorithm': 'kmeans', 'clusters': {}}
    
    async def _apply_dbscan_clustering(self, X: np.ndarray, phone_numbers: np.ndarray) -> Dict:
        """
        Apply DBSCAN clustering
        """
        try:
            dbscan = DBSCAN(**self.algorithm_params['dbscan'])
            labels = dbscan.fit_predict(X)
            
            # Organize clusters
            clusters = defaultdict(list)
            noise_points = []
            
            for phone, label in zip(phone_numbers, labels):
                if label == -1:
                    noise_points.append(phone)
                else:
                    clusters[label].append(phone)
            
            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            silhouette = 0
            if n_clusters > 1:
                silhouette = silhouette_score(X, labels)
            
            return {
                'algorithm': 'dbscan',
                'clusters': dict(clusters),
                'noise_points': noise_points,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette_score': silhouette
            }
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {str(e)}")
            return {'algorithm': 'dbscan', 'clusters': {}}
    
    async def _calculate_feature_clustering_metrics(self, X: np.ndarray, clustering_results: Dict) -> Dict:
        """
        Calculate metrics for feature-based clustering
        """
        metrics = {}
        
        for algorithm, result in clustering_results.items():
            if 'clusters' not in result or not result['clusters']:
                continue
            
            algorithm_metrics = {
                'n_clusters': result.get('n_clusters', 0),
                'silhouette_score': result.get('silhouette_score', 0)
            }
            
            # Calculate cluster size statistics
            cluster_sizes = [len(members) for members in result['clusters'].values()]
            if cluster_sizes:
                algorithm_metrics['avg_cluster_size'] = np.mean(cluster_sizes)
                algorithm_metrics['max_cluster_size'] = max(cluster_sizes)
                algorithm_metrics['min_cluster_size'] = min(cluster_sizes)
            
            metrics[algorithm] = algorithm_metrics
        
        return metrics
    
    async def _analyze_feature_importance(self, X: np.ndarray, feature_names: List[str], 
                                        clustering_results: Dict) -> Dict:
        """
        Analyze which features are most important for clustering
        """
        try:
            # This is a simplified feature importance analysis
            # In practice, you might use more sophisticated methods
            
            feature_importance = {}
            
            for algorithm, result in clustering_results.items():
                if 'clusters' not in result or not result['clusters']:
                    continue
                
                # Calculate feature variance within vs between clusters
                if algorithm == 'kmeans' and 'cluster_centers' in result:
                    centers = result['cluster_centers']
                    
                    # Calculate between-cluster variance for each feature
                    between_var = np.var(centers, axis=0)
                    
                    # Normalize to get importance scores
                    importance_scores = between_var / (np.sum(between_var) + 1e-6)
                    
                    feature_importance[algorithm] = dict(zip(feature_names, importance_scores))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {str(e)}")
            return {}