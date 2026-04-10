import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from pathlib import Path

def train_random_forest(df, dataset_name="IoT"):
    """
    Train Random Forest model
    """
    print("="*60)
    print(f"🌲 RANDOM FOREST - {dataset_name}")
    print("="*60)
    
    # Prepare data
    X = df.drop('outcome', axis=1, errors='ignore')
    y = df['outcome']
    
    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize (optional for RF but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 Training samples: {X_train.shape[0]}")
    print(f"📊 Testing samples: {X_test.shape[0]}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("📊 MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create output directory
    output_dir = Path("outputs/models/random_forest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'{dataset_name} - Random Forest Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(output_dir / f'{dataset_name.lower()}_confusion.png', dpi=150)
    plt.close()
    print("✅ Confusion matrix saved")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / f'{dataset_name.lower()}_roc.png', dpi=150)
    plt.close()
    print("✅ ROC curve saved")
    
    # 3. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title(f'{dataset_name} - Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name.lower()}_importance.png', dpi=150)
    plt.close()
    print("✅ Feature importance saved")
    
    # Save metrics and importance
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    pd.DataFrame([metrics]).to_csv(output_dir / f'{dataset_name.lower()}_metrics.csv', index=False)
    feature_importance.to_csv(output_dir / f'{dataset_name.lower()}_feature_importance.csv', index=False)
    
    print(f"\n✅ Random Forest complete for {dataset_name}")
    return model, scaler, metrics

if __name__ == "__main__":
    import seaborn as sns
    
    data_dir = Path("data/processed")
    
    # MQTT
    df_mqtt = pd.read_csv(data_dir / "mqtt_cleaned.csv")
    model_mqtt, scaler_mqtt, metrics_mqtt = train_random_forest(df_mqtt, "MQTT")
    
    # UDP
    df_udp = pd.read_csv(data_dir / "udp_cleaned.csv")
    model_udp, scaler_udp, metrics_udp = train_random_forest(df_udp, "UDP")