import pandas as pd
import os
from sklearn.utils import resample

def load_and_combine_data():
    raw_files = {
        "Jamaican": "data/raw/jamaican.csv",
        "Nigerian": "data/raw/nigerian.csv", 
        "Scottish": "data/raw/scottish.csv",
        "Australian": "data/raw/australian.csv",
        "Southern US": "data/raw/southern_us.csv"
    }
    
    synthetic = pd.read_csv("data/synthetic/all_dialects.csv")
    
    dfs = []
    for dialect, path in raw_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df[["text", "label"]]) 
    
    dfs.append(synthetic[["text", "label"]])
    combined = pd.concat(dfs)
    return combined

def balance_dataset(df, samples_per_class=100):
    balanced_dfs = []
    for dialect in df['label'].unique():
        dialect_df = df[df['label'] == dialect]
        balanced_df = resample(dialect_df,
                             replace=len(dialect_df) < samples_per_class,
                             n_samples=samples_per_class,
                             random_state=42)
        balanced_dfs.append(balanced_df)
    
    return pd.concat(balanced_dfs)

def save_datasets(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/full_unbalanced.csv", index=False)
    balanced = balance_dataset(df)
    balanced.to_csv("data/processed/balanced.csv", index=False)

if __name__ == "__main__":
    combined_data = load_and_combine_data()
    save_datasets(combined_data)
    print("Saved to:")
    print("data/processed/full_unbalanced.csv")
    print("data/processed/balanced.csv")