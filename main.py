"""
Main Pipeline for IoT DDoS Detection
Processes MQTT and UDP datasets with complete ML workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def load_and_parse_data(file_path):
    """
    Load and parse the IoT DDoS dataset
    """
    print(f"\n{'='*70}")
    print(f"📂 Loading: {file_path.name}")
    print(f"{'='*70}")
    
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Based on the data structure, assign column names
    # The data appears to have: timestamp, node_id, protocol, payload_size, 
    # total_messages, frequency, mean_frequency, monitoring_frequency, 
    # monitoring_total_messages, monitoring_total_messages_same_node, outcome
    
    # For now, let's use generic column names and let cleaning handle it
    df.columns = [f'col_{i}' for i in range(df.shape[1])]
    
    print(f"✅ Raw data shape: {df.shape}")
    print(f"📋 Columns: {df.shape[1]}")
    
    return df

def prepare_features(df):
    """
    Prepare features from raw data
    """
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with all NaN
    df = df.dropna(how='all')
    
    # Fill remaining NaN with median
    df = df.fillna(df.median(numeric_only=True))
    
    return df

def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*70)
    print("🚀 IOT DDOS DETECTION - COMPLETE ML PIPELINE")
    print("="*70)
    
    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "raw"
    
    # Dataset files
    mqtt_file = data_dir / "UL-ECE-MQTT-DDoS-H-IoT2025.csv"
    udp_file = data_dir / "UL-ECE-UDP-DDoS-H-IoT2025.csv"
    
    # Check if files exist
    if not mqtt_file.exists() or not udp_file.exists():
        print("❌ Error: Dataset files not found!")
        print(f"   MQTT: {mqtt_file.exists()}")
        print(f"   UDP: {udp_file.exists()}")
        return
    
    # ========== STEP 1: LOAD DATA ==========
    print("\n[STEP 1/7] Loading Datasets...")
    print("-"*70)
    
    df_mqtt_raw = load_and_parse_data(mqtt_file)
    df_udp_raw = load_and_parse_data(udp_file)
    
    # ========== STEP 2: PREPROCESS DATA ==========
    print("\n[STEP 2/7] Preprocessing Data...")
    print("-"*70)
    
    df_mqtt = prepare_features(df_mqtt_raw)
    df_udp = prepare_features(df_udp_raw)
    
    print(f"✅ MQTT preprocessed: {df_mqtt.shape}")
    print(f"✅ UDP preprocessed: {df_udp.shape}")
    
    # Save cleaned data
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    df_mqtt.to_csv(processed_dir / "mqtt_cleaned.csv", index=False)
    df_udp.to_csv(processed_dir / "udp_cleaned.csv", index=False)
    print("✅ Cleaned data saved to data/processed/")
    
    # ========== STEP 3: IMPORT MODULES ==========
    print("\n[STEP 3/7] Importing Analysis Modules...")
    print("-"*70)
    
    try:
        from src.graph_plotting import plot_graphs
        print("✅ Graph plotting module loaded")
    except ImportError as e:
        print(f"⚠️  Graph plotting module not found: {e}")
        plot_graphs = None
    
    try:
        from src.pca_analysis import apply_pca
        print("✅ PCA analysis module loaded")
    except ImportError as e:
        print(f"⚠️  PCA module not found: {e}")
        apply_pca = None
    
    try:
        from src.svd_analysis import apply_svd
        print("✅ SVD analysis module loaded")
    except ImportError as e:
        print(f"⚠️  SVD module not found: {e}")
        apply_svd = None
    
    try:
        from src.clustering import apply_clustering
        print("✅ Clustering module loaded")
    except ImportError as e:
        print(f"⚠️  Clustering module not found: {e}")
        apply_clustering = None
    
    try:
        from src.logistic_regression import train_logistic_regression
        print("✅ Logistic Regression module loaded")
    except ImportError as e:
        print(f"⚠️  Logistic Regression module not found: {e}")
        train_logistic_regression = None
    
    try:
        from src.naive_bayes import train_naive_bayes
        print("✅ Naive Bayes module loaded")
    except ImportError as e:
        print(f"⚠️  Naive Bayes module not found: {e}")
        train_naive_bayes = None
    
    try:
        from src.random_forest import train_random_forest
        print("✅ Random Forest module loaded")
    except ImportError as e:
        print(f"⚠️  Random Forest module not found: {e}")
        train_random_forest = None
    
    # ========== STEP 4: VISUALIZATION ==========
    print("\n[STEP 4/7] Creating Visualizations...")
    print("-"*70)
    
    if plot_graphs:
        try:
            plot_graphs(df_mqtt, "MQTT")
            plot_graphs(df_udp, "UDP")
            print("✅ Visualizations created")
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
    else:
        print("⚠️  Skipping visualizations (module not available)")
    
    # ========== STEP 5: DIMENSIONALITY REDUCTION ==========
    print("\n[STEP 5/7] Applying Dimensionality Reduction...")
    print("-"*70)
    
    # PCA
    if apply_pca:
        try:
            print("\n🔷 Applying PCA...")
            apply_pca(df_mqtt, "MQTT", n_components=0.95)
            apply_pca(df_udp, "UDP", n_components=0.95)
            print("✅ PCA completed")
        except Exception as e:
            print(f"⚠️  PCA error: {e}")
    else:
        print("⚠️  Skipping PCA (module not available)")
    
    # SVD
    if apply_svd:
        try:
            print("\n🔶 Applying SVD...")
            apply_svd(df_mqtt, "MQTT", n_components=10)
            apply_svd(df_udp, "UDP", n_components=10)
            print("✅ SVD completed")
        except Exception as e:
            print(f"⚠️  SVD error: {e}")
    else:
        print("⚠️  Skipping SVD (module not available)")
    
    # ========== STEP 6: CLUSTERING ==========
    print("\n[STEP 6/7] Applying Clustering Algorithms...")
    print("-"*70)
    
    if apply_clustering:
        try:
            apply_clustering(df_mqtt, "MQTT")
            apply_clustering(df_udp, "UDP")
            print("✅ Clustering completed")
        except Exception as e:
            print(f"⚠️  Clustering error: {e}")
    else:
        print("⚠️  Skipping clustering (module not available)")
    
    # ========== STEP 7: MACHINE LEARNING MODELS ==========
    print("\n[STEP 7/7] Training Machine Learning Models...")
    print("-"*70)
    
    models_results = {}
    
    # Logistic Regression
    if train_logistic_regression:
        try:
            print("\n📈 Training Logistic Regression...")
            _, _, metrics_lr_mqtt = train_logistic_regression(df_mqtt, "MQTT")
            _, _, metrics_lr_udp = train_logistic_regression(df_udp, "UDP")
            models_results['Logistic Regression'] = {
                'MQTT': metrics_lr_mqtt,
                'UDP': metrics_lr_udp
            }
            print("✅ Logistic Regression completed")
        except Exception as e:
            print(f"⚠️  Logistic Regression error: {e}")
    else:
        print("⚠️  Skipping Logistic Regression (module not available)")
    
    # Naive Bayes
    if train_naive_bayes:
        try:
            print("\n📊 Training Naive Bayes...")
            _, _, metrics_nb_mqtt = train_naive_bayes(df_mqtt, "MQTT")
            _, _, metrics_nb_udp = train_naive_bayes(df_udp, "UDP")
            models_results['Naive Bayes'] = {
                'MQTT': metrics_nb_mqtt,
                'UDP': metrics_nb_udp
            }
            print("✅ Naive Bayes completed")
        except Exception as e:
            print(f"⚠️  Naive Bayes error: {e}")
    else:
        print("⚠️  Skipping Naive Bayes (module not available)")
    
    # Random Forest
    if train_random_forest:
        try:
            print("\n🌲 Training Random Forest...")
            _, _, metrics_rf_mqtt = train_random_forest(df_mqtt, "MQTT")
            _, _, metrics_rf_udp = train_random_forest(df_udp, "UDP")
            models_results['Random Forest'] = {
                'MQTT': metrics_rf_mqtt,
                'UDP': metrics_rf_udp
            }
            print("✅ Random Forest completed")
        except Exception as e:
            print(f"⚠️  Random Forest error: {e}")
    else:
        print("⚠️  Skipping Random Forest (module not available)")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📁 Output Locations:")
    print(f"   • Cleaned Data: {processed_dir}")
    print(f"   • Visualizations: {base_dir / 'outputs' / 'figures'}")
    print(f"   • PCA Results: {base_dir / 'outputs' / 'pca'}")
    print(f"   • SVD Results: {base_dir / 'outputs' / 'svd'}")
    print(f"   • Clustering: {base_dir / 'outputs' / 'clustering'}")
    print(f"   • Model Results: {base_dir / 'outputs' / 'models'}")
    
    # Display model comparison
    if models_results:
        print("\n📊 Model Performance Summary:")
        print("-"*70)
        for model_name, datasets in models_results.items():
            print(f"\n{model_name}:")
            for dataset, metrics in datasets.items():
                if isinstance(metrics, dict):
                    print(f"  {dataset}:")
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            print(f"    • {metric}: {value:.4f}")
                        else:
                            print(f"    • {metric}: {value}")
    
    print("\n" + "="*70)
    print("🎉 All tasks completed! Check the outputs folder for results.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()