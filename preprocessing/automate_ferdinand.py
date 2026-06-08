import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def run_preprocessing(input_path, output_path):
    print(f"[*] Membaca data mentah dari: {input_path}")
    df = pd.read_csv(input_path)
    
   
    df.drop_duplicates(inplace=True)
    
   
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
   
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Data preprocessing selesai! Disimpan di: {output_path}")

if __name__ == '__main__':
    raw_path = 'namadataset_raw/Credit Risk Benchmark Dataset.csv'
    clean_path = 'namadataset_preprocessing/Credit_Risk_Clean.csv'
    run_preprocessing(raw_path, clean_path)