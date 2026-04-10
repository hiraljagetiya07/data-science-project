import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path

def apply_clustering(df, dataset_name="IoT"):
    """
    Apply clustering techniques
    """
    print("="*60)
    print(f"🔵 CLUSTERING ANALYSIS - {dataset_name}")
    print("="*60)
    
    # Prepare data
    X = df.drop('outcome', axis=1, errors='ignore')
    y_true = df['outcome'] if 'outcome' in df.columns else None
    
    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create output directory
    output_dir = Path("outputs/clustering")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. K-Means Clustering
    print("\n📊 Applying K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    
    # Elbow method
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        inertias.append(kmeans_temp.inertia_)
    
    plt.figure()
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title(f'{dataset_name} - Elbow Method for K-Means')
    plt.grid(True)
    plt.savefig(output_dir / f'{dataset_name.lower()}_elbow.png', dpi=150)
    plt.close()
    print("✅ Elbow plot saved")
    
    # K-Means visualization
    if X_scaled.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_kmeans, 
                   cmap='viridis', alpha=0.6, edgecolors='k')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', s=200, marker='X', label='Centroids')
        plt.title(f'{dataset_name} - K-Means Clustering')
        plt.legend()
        plt.savefig(output_dir / f'{dataset_name.lower()}_kmeans.png', dpi=150)
        plt.close()
    
    if y_true is not None:
        sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
        results['K-Means'] = {'silhouette': sil_kmeans, 'labels': labels_kmeans}
        print(f"✅ K-Means Silhouette Score: {sil_kmeans:.4f}")
    
    # 2. Hierarchical Clustering
    print("\n📊 Applying Hierarchical Clustering...")
    hierarchical = AgglomerativeClustering(n_clusters=2)
    labels_hier = hierarchical.fit_predict(X_scaled)
    
    if y_true is not None:
        sil_hier = silhouette_score(X_scaled, labels_hier)
        results['Hierarchical'] = {'silhouette': sil_hier, 'labels': labels_hier}
        print(f"✅ Hierarchical Silhouette Score: {sil_hier:.4f}")
    
    # 3. DBSCAN
    print("\n📊 Applying DBSCAN...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_scaled)
    
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    print(f"✅ DBSCAN found {n_clusters_dbscan} clusters")
    
    if n_clusters_dbscan > 1 and y_true is not None:
        sil_dbscan = silhouette_score(X_scaled, labels_dbscan)
        results['DBSCAN'] = {'silhouette': sil_dbscan, 'labels': labels_dbscan}
        print(f"✅ DBSCAN Silhouette Score: {sil_dbscan:.4f}")
    
    # Save results
    df_results = pd.DataFrame({
        'kmeans_labels': labels_kmeans,
        'hierarchical_labels': labels_hier,
        'dbscan_labels': labels_dbscan
    })
    df_results.to_csv(output_dir / f'{dataset_name.lower()}_clusters.csv', index=False)
    print(f"\n✅ Cluster labels saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    data_dir = Path("data/processed")
    
    # MQTT
    df_mqtt = pd.read_csv(data_dir / "mqtt_cleaned.csv")
    results_mqtt = apply_clustering(df_mqtt, "MQTT")
    
    # UDP
    df_udp = pd.read_csv(data_dir / "udp_cleaned.csv")
    results_udp = apply_clustering(df_udp, "UDP")