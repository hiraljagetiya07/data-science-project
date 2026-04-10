import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_graphs(df, dataset_name="IoT"):
    """
    Create comprehensive visualizations
    """
    print("="*60)
    print(f"📊 CREATING VISUALIZATIONS - {dataset_name}")
    print("="*60)
    
    # Create output directory
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [10, 6]
    
    # 1. Target Distribution
    plt.figure()
    sns.countplot(data=df, x='outcome', palette='viridis')
    plt.title(f'{dataset_name} - Attack vs Normal Traffic Distribution')
    plt.xlabel('Outcome (0=Normal, 1=Attack)')
    plt.ylabel('Count')
    plt.savefig(output_dir / f'{dataset_name.lower()}_target_dist.png', dpi=150)
    plt.close()
    print("✅ Target distribution plot saved")
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title(f'{dataset_name} - Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name.lower()}_correlation.png', dpi=150)
    plt.close()
    print("✅ Correlation heatmap saved")
    
    # 3. Feature Distributions
    numeric_cols = df.select_dtypes(include=['number']).columns[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            sns.histplot(data=df, x=col, ax=axes[idx], kde=True, color='skyblue')
            axes[idx].set_title(f'{col} Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name.lower()}_distributions.png', dpi=150)
    plt.close()
    print("✅ Feature distributions saved")
    
    # 4. Box Plot by Outcome
    numeric_cols = df.select_dtypes(include=['number']).columns[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes) and col != 'outcome':
            sns.boxplot(data=df, x='outcome', y=col, ax=axes[idx], palette='Set2')
            axes[idx].set_title(f'{col} by Attack Status')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name.lower()}_boxplot.png', dpi=150)
    plt.close()
    print("✅ Box plots saved")
    
    # 5. Pair Plot (sample)
    sample_df = df.sample(min(1000, len(df)))
    pair_cols = ['frequency', 'payload_size', 'total_messages', 'outcome']
    pair_cols = [col for col in pair_cols if col in sample_df.columns]
    
    if len(pair_cols) >= 2:
        sns.pairplot(sample_df, vars=pair_cols[:-1], hue='outcome', 
                    palette='viridis', plot_kws={'alpha': 0.6})
        plt.suptitle(f'{dataset_name} - Pair Plot', y=1.02)
        plt.savefig(output_dir / f'{dataset_name.lower()}_pairplot.png', dpi=150)
        plt.close()
        print("✅ Pair plot saved")
    
    print(f"\n✅ All visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Plot for both datasets
    data_dir = Path("data/processed")
    
    df_mqtt = pd.read_csv(data_dir / "mqtt_cleaned.csv")
    plot_graphs(df_mqtt, "MQTT")
    
    df_udp = pd.read_csv(data_dir / "udp_cleaned.csv")
    plot_graphs(df_udp, "UDP")