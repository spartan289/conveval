import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
import json
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from kneed import KneeLocator
import os
from dotenv import load_dotenv
load_dotenv()

class ConversationInsightClusterer:
    def __init__(self, api_key):
        self.insights = []
        self.client = OpenAI(api_key=api_key)
        self.clusters = {}
        self.feature_matrix = None
        self.cluster_labels = None
        
    def load_insights(self, insights_list):
        """Load insights from a list"""
        self.insights = insights_list
        print(f"Loaded {len(insights_list)} insights")
        
    def load_from_file(self, filepath):
        """Load insights from various file formats"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            # Assume insights are in first column or column named 'insight'
            if 'insight' in df.columns:
                self.insights = df['insight'].tolist()
            else:
                self.insights = df.iloc[:, 0].tolist()
        elif filepath.endswith('.txt'):
            with open(filepath, 'r') as f:
                self.insights = [line.strip() for line in f if line.strip()]
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.insights = data
                elif isinstance(data, dict) and 'insights' in data:
                    self.insights = data['insights']
        print(f"Loaded {len(self.insights)} insights from {filepath}")
    
    def generate_embeddings(self, text_list, batch_size=100):
        all_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i:i + batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)
    
    def preprocess_insights(self):
        print("Generating embeddings...")
        self.feature_matrix = self.generate_embeddings(self.insights)
        print(f"Generated embeddings with shape: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def kmeans_clustering(self, n_clusters=5):
        """Perform K-means clustering"""
        if self.feature_matrix is None:
            self.preprocess_insights()
            
        print(f"Running K-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.feature_matrix)
        
        self.cluster_labels = labels
        self.kmeans_model = kmeans
        
        sil_score = silhouette_score(self.feature_matrix, labels)
        print(f"Silhouette Score: {sil_score:.3f}")
        #labels count
        return labels
    

    def visualize_clusters(self, method='kmeans', save_path=None):
        if self.feature_matrix is None:
            print("No clustering data available")
            return
            
        print("Creating TSNE visualization of clusters...")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, n_iter=300, perplexity=5)
        print("Visulaizing Cluster")
        plt.figure(figsize=(12, 8))
        reduced_features = tsne.fit_transform(self.feature_matrix)
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=self.cluster_labels, cmap='viridis', alpha=0.6)
        plt.title(f'Cluster Visualization ({method.upper()})')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.colorbar(label='Cluster Label')
        plt.grid(True)
        plt.show()
    
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using silhouette score"""
        if self.feature_matrix is None:
            self.preprocess_insights()
            
        print("Finding optimal number of clusters...")
        scores = []
        cluster_range = range(2, min(max_clusters + 1, len(self.insights)))
        
        for n_clusters in tqdm(cluster_range):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.feature_matrix)
            score = silhouette_score(self.feature_matrix, labels)
            scores.append(score)
        
        optimal_clusters = cluster_range[np.argmax(scores)]
        print(f"\nOptimal number of clusters: {optimal_clusters}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, scores, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Optimal Number of Clusters (K-Means)')
        plt.grid(True)
        plt.axvline(x=optimal_clusters, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_clusters}')
        plt.legend()
        plt.show()
        
        return optimal_clusters
    def find_optimal_clusters_elbow(self, optimal_k, max_clusters=100):
        """Find optimal number of clusters using elbow method (inertia)"""
        if self.feature_matrix is None:
            self.preprocess_insights()

        print("Finding optimal number of clusters using elbow method...")
        inertias = []
        cluster_range = range(1, max_clusters)

        for k in tqdm(cluster_range):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.feature_matrix)
            inertias.append(kmeans.inertia_)

        # Automatically find the elbow point
        kl = KneeLocator(list(cluster_range), inertias, curve="convex", direction="decreasing")
        optimal_k = kl.elbow

        # Plot the elbow graph
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, inertias, 'bo-')
        if optimal_k:
            plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Elbow at k={optimal_k}')
            plt.legend()
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia (Within-Cluster SSE)')
        plt.title('Elbow Method for Optimal Clusters')
        plt.grid(True)
        plt.show()

        # print(f"\nOptimal number of clusters (Elbow): {optimal_k}")
        return optimal_k    
    def predict_cluster(self, new_insight):
        """Predict cluster for a new insight"""
        if not hasattr(self, 'kmeans_model'):
            print("No trained model found. Please run kmeans_clustering first.")
            return None
            
        # Generate embedding for new insight
        embedding = self.generate_embeddings([new_insight])
        
        # Predict cluster
        cluster = self.kmeans_model.predict(embedding)[0]
        
        print(f"New insight assigned to cluster: {cluster}")
        return cluster


def cluster_insights(transcripts_path: str, n_cluster=5, save_data=True)-> list:
    """
    Cluster conversation insights from transcripts file.
    """
    with open(transcripts_path, 'r') as f:
        transcripts = json.load(f)

    insights = []
    for transcript in transcripts:
        
        insights.append(transcript['insights'])
    
    clusterer = ConversationInsightClusterer(os.getenv('OPENAI_API_KEY'))
    clusterer.load_insights(insights)
    clusterer.preprocess_insights()
    
    print("=== K-MEANS CONVERSATION INSIGHT CLUSTERING ===\n")
    
    print("Step 1: Finding optimal number of clusters...")
    optimal_k = clusterer.find_optimal_clusters_elbow(optimal_k=n_cluster, max_clusters=min(40,len(transcripts)))
    print(f"\nStep 2: Running K-means with {n_cluster} clusters...")
    cluster_labels= clusterer.kmeans_clustering(n_clusters=n_cluster)
    label_counts = Counter(cluster_labels)
    print("Cluster label counts:")
    for label, count in label_counts.items():
        print(f"Cluster {label}: {count} insights")

    clusterer.visualize_clusters('kmeans')
    
    for ind, transcript in enumerate(transcripts):
        transcript["cluster_number"]=int(cluster_labels[ind])

    if save_data:
        with open(transcripts_path, 'w') as f:
            json.dump(transcripts, f, indent=4, ensure_ascii=False)
    print(f"\nTranscripts updated with cluster labels and saved to {transcripts_path}")
    return transcripts
    

def cluster_insights_with_list(texts: list, n_cluster: int, max_cluster=100):
    """
    Cluster conversation insights from transcripts file.
    """
    
    clusterer = ConversationInsightClusterer(os.getenv('OPENAI_API_KEY'))
    clusterer.load_insights(texts)
    clusterer.preprocess_insights()
    
    print("=== K-MEANS CONVERSATION INSIGHT CLUSTERING ===\n")
    
    print("Step 1: Finding optimal number of clusters...")
    print(f"\nStep 2: Running K-means with {n_cluster} clusters...")
    cluster_labels= clusterer.kmeans_clustering(n_clusters=n_cluster)
    clusterer.find_optimal_clusters_elbow(optimal_k=n_cluster, max_clusters=max_cluster)

    clusterer.visualize_clusters('kmeans')
    


    # create cluster dictionary
    cluster_dict = {}
    for ind, labels in enumerate(cluster_labels):
        if labels not in cluster_dict:
            cluster_dict[labels]=[]
        cluster_dict[labels].append(texts[ind])
    return cluster_dict

