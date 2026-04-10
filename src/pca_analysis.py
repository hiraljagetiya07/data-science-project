import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def apply_pca(df, dataset_name="IoT", n_components=0.95):
    """
    Apply PCA for dimensionality reduction
    """
    print("="*60)
    print(f"🔷 PCA ANALYSIS - {dataset_name}")
    print("="*60)
    
    # Prepare data
    X = df.drop('outcome', axis=1, errors='ignore')
    y = df['outcome'] if 'outcome' in df.columns else None
    
    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n📊 Original features: {X_scaled.shape[1]}")
    
    # Apply PCA
    if isinstance(n_components, float):
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)
    
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"✅ Reduced features: {X_pca.shape[1]}")
    print(f"📈 Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"📊 Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Create output directory
    output_dir = Path("outputs/pca")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Scree Plot
    plt.figure()
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
             np.cumsum(pca.explained_variance_ratio_), 'b-o', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'{dataset_name} - PCA Scree Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f'{dataset_name.lower()}_scree.png', dpi=150)
    plt.close()
    print("✅ Scree plot saved")
    
    # 2. 2D Projection
    if X_pca.shape[1] >= 2:
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        if y is not None:
            scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                                 c=y, cmap='viridis', alpha=0.6, edgecolors='k')
            plt.colorbar(scatter, label='Outcome')
        else:
            plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.6)
        
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'{dataset_name} - PCA 2D Projection')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{dataset_name.lower()}_pca_2d.png', dpi=150)
        plt.close()
        print("✅ 2D projection saved")
    
    # 3. Feature Importance (Loadings)
    if hasattr(pca, 'components_'):
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
            index=X.columns
        )
        loadings.to_csv(output_dir / f'{dataset_name.lower()}_loadings.csv')
        print("✅ Feature loadings saved")
    
    print(f"\n✅ PCA analysis complete for {dataset_name}")
    return X_pca, pca, scaler

if __name__ == "__main__":
    data_dir = Path("data/processed")
    
    # MQTT
    df_mqtt = pd.read_csv(data_dir / "mqtt_cleaned.csv")
    X_pca_mqtt, pca_mqtt, scaler_mqtt = apply_pca(df_mqtt, "MQTT")
    
    # UDP
    df_udp = pd.read_csv(data_dir / "udp_cleaned.csv")
    X_pca_udp, pca_udp, scaler_udp = apply_pca(df_udp, "UDP")