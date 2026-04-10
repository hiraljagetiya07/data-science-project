import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def apply_svd(df, dataset_name="IoT", n_components=10):
    """
    Apply SVD for dimensionality reduction
    """
    print("="*60)
    print(f"🔶 SVD ANALYSIS - {dataset_name}")
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
    
    # Apply SVD
    n_comp = min(n_components, X_scaled.shape[1]-1, X_scaled.shape[0]-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_svd = svd.fit_transform(X_scaled)
    
    print(f"✅ Reduced features: {X_svd.shape[1]}")
    print(f"📈 Explained variance ratio: {svd.explained_variance_ratio_}")
    print(f"📊 Total variance explained: {svd.explained_variance_ratio_.sum():.4f}")
    
    # Create output directory
    output_dir = Path("outputs/svd")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Variance Plot
    plt.figure()
    plt.bar(range(1, len(svd.explained_variance_ratio_)+1), 
            svd.explained_variance_ratio_, alpha=0.7, color='green')
    plt.plot(range(1, len(svd.explained_variance_ratio_)+1), 
             np.cumsum(svd.explained_variance_ratio_), 'r-o', linewidth=2)
    plt.xlabel('Number of SVD Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'{dataset_name} - SVD Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'{dataset_name.lower()}_svd_variance.png', dpi=150)
    plt.close()
    print("✅ Variance plot saved")
    
    # 2. 2D Projection
    if X_svd.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        if y is not None:
            scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1], 
                                 c=y, cmap='viridis', alpha=0.6, edgecolors='k')
            plt.colorbar(scatter, label='Outcome')
        else:
            plt.scatter(X_svd[:, 0], X_svd[:, 1], alpha=0.6)
        
        plt.xlabel(f'SVD1 ({svd.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'SVD2 ({svd.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'{dataset_name} - SVD 2D Projection')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{dataset_name.lower()}_svd_2d.png', dpi=150)
        plt.close()
        print("✅ 2D projection saved")
    
    print(f"\n✅ SVD analysis complete for {dataset_name}")
    return X_svd, svd, scaler

if __name__ == "__main__":
    data_dir = Path("data/processed")
    
    # MQTT
    df_mqtt = pd.read_csv(data_dir / "mqtt_cleaned.csv")
    X_svd_mqtt, svd_mqtt, scaler_mqtt = apply_svd(df_mqtt, "MQTT", n_components=10)
    
    # UDP
    df_udp = pd.read_csv(data_dir / "udp_cleaned.csv")
    X_svd_udp, svd_udp, scaler_udp = apply_svd(df_udp, "UDP", n_components=10)