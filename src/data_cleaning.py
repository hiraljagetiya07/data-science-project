import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(file_path):
    """
    Clean IoT DDoS dataset
    """
    print("="*60)
    print("🧹 DATA CLEANING PROCESS")
    print("="*60)
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"\n📊 Original shape: {df.shape}")
    
    # 1. Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.fillna(df.median(numeric_only=True))
    print(f"✅ Missing values filled: {missing_before}")
    
    # 2. Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"✅ Duplicates removed: {duplicates}")
    
    # 3. Remove irrelevant columns (IP addresses, descriptions)
    cols_to_drop = ['source_ip', 'source_ip_des', 'destination_ip', 
                    'destination_ip_des', 'protocol_des']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"✅ Dropped {len(cols_to_drop)} irrelevant columns")
    
    # 4. Convert protocol to numeric if needed
    if 'protocol' in df.columns:
        df['protocol'] = pd.to_numeric(df['protocol'], errors='coerce')
    
    # 5. Handle outliers using IQR
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'outcome':  # Don't modify target
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    print("✅ Outliers handled using IQR method")
    
    # 6. Final shape
    print(f"\n📊 Cleaned shape: {df.shape}")
    print(f"✅ Data cleaning complete!")
    
    return df

if __name__ == "__main__":
    # Load and clean both datasets
    data_dir = Path("data/raw")
    
    # Clean MQTT dataset
    print("\n🔵 Cleaning MQTT Dataset...")
    df_mqtt = clean_data(data_dir / "UL-ECE-MQTT-DDoS-H-IoT2025.csv")
    df_mqtt.to_csv("data/processed/mqtt_cleaned.csv", index=False)
    
    # Clean UDP dataset
    print("\n🟢 Cleaning UDP Dataset...")
    df_udp = clean_data(data_dir / "UL-ECE-UDP-DDoS-H-IoT2025.csv")
    df_udp.to_csv("data/processed/udp_cleaned.csv", index=False)
    
    print("\n✅ Cleaned data saved to data/processed/")