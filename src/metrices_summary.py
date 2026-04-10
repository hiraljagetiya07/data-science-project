import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def generate_summary():
    """
    Generate summary of all model metrics
    """
    print("="*60)
    print("📊 GENERATING METRICS SUMMARY")
    print("="*60)
    
    output_dir = Path("outputs/models")
    summary_data = []
    
    # Collect metrics from all models
    for dataset in ['MQTT', 'UDP']:
        for model in ['logistic_regression', 'naive_bayes', 'random_forest']:
            metrics_file = output_dir / model / f"{dataset.lower()}_metrics.csv"
            
            if metrics_file.exists():
                df_metrics = pd.read_csv(metrics_file)
                df_metrics['Dataset'] = dataset
                df_metrics['Model'] = model.replace('_', ' ').title()
                summary_data.append(df_metrics)
    
    # Combine all metrics
    if summary_data:
        df_summary = pd.concat(summary_data, ignore_index=True)
        
        # Display summary
        print("\n" + "="*60)
        print("📋 MODEL COMPARISON")
        print("="*60)
        print(df_summary[['Dataset', 'Model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))
        
        # Save summary
        df_summary.to_csv("outputs/models/summary.csv", index=False)
        print(f"\n✅ Summary saved to outputs/models/summary.csv")
        
        # Comparison plot
        plt.figure(figsize=(12, 6))
        x = range(len(df_summary))
        width = 0.15
        
        plt.bar([i - width*2 for i in x], df_summary['accuracy'], width, label='Accuracy')
        plt.bar([i - width for i in x], df_summary['precision'], width, label='Precision')
        plt.bar(x, df_summary['recall'], width, label='Recall')
        plt.bar([i + width for i in x], df_summary['f1_score'], width, label='F1-Score')
        plt.bar([i + width*2 for i in x], df_summary['roc_auc'], width, label='ROC-AUC')
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, [f"{row['Dataset']}\n{row['Model']}" for _, row in df_summary.iter()], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/models/comparison.png", dpi=150)
        plt.close()
        print("✅ Comparison plot saved")
        
        return df_summary
    else:
        print("❌ No metrics files found!")
        return None

if __name__ == "__main__":
    summary = generate_summary()