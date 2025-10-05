import requests
import time
import os
import zipfile
import traceback
import pandas as pd
import shutil
import glob

urls = [
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
]

filenames = [
    "heart_disease.zip"
]

save_dir = "data/raw"

os.makedirs(save_dir, exist_ok=True)

for url, filename in zip(urls, filenames):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            path = os.path.join(save_dir, filename)
            with open(path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")

            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            print(f"Extracted: {filename}")

            os.remove(path)
            print(f"Deleted ZIP file: {filename}")

            cleveland_file = os.path.join(save_dir, 'processed.cleveland.data')
            if os.path.exists(cleveland_file):
                print(f"Found processed.cleveland.data file.")
                
                print(f"Processing: processed.cleveland.data")
                df_temp = None
                try:
                    df_temp = pd.read_csv(cleveland_file, sep=',', encoding='utf-8', header=None, na_values='?')
                    if df_temp.shape[1] != 14:
                        raise ValueError(f"Expected 14 columns, got {df_temp.shape[1]}")
                    print(f"  - Read as CSV. Shape: {df_temp.shape}")
                except Exception as e1:
                    print(f"  - CSV read failed: {e1}")
                    try:
                        df_temp = pd.read_csv(cleveland_file, sep=r'\s+', header=None, encoding='latin-1', na_values='?')
                        if df_temp.shape[1] != 14:
                            raise ValueError(f"Expected 14 columns, got {df_temp.shape[1]}")
                        print(f"  - Read as space-separated. Shape: {df_temp.shape}")
                    except Exception as e2:
                        print(f"  - Space-separated read failed: {e2}")
                
                if df_temp is not None and df_temp.shape[1] == 14:
                    csv_path = os.path.join(save_dir, 'cleveland.csv')
                    df_temp.to_csv(csv_path, index=False, header=False)
                    print(f"\nCreated CSV: {csv_path}")
                    print(f"Dataset shape: {df_temp.shape} (rows, columns)")
                    print(f"Columns: {list(df_temp.columns)}")
                    
                    print(f"Missing values per column:\n{df_temp.isnull().sum()}")

                    all_files = glob.glob(os.path.join(save_dir, '*'))
                    for file_path in all_files:
                        if os.path.isfile(file_path) and not file_path.endswith('.csv'):
                            os.remove(file_path)
                            print(f"Deleted file: {os.path.basename(file_path)}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            print(f"Deleted folder: {os.path.basename(file_path)}")
                    
                    print("Cleanup completed. Only cleveland.csv remains in data/raw.")
                else:
                    print("Failed to process processed.cleveland.data: Invalid shape.")
            else:
                print("Warning: processed.cleveland.data file not found after extraction.")
        
        else:
            print(f"Failed to download {filename}: {response.status_code}")
        
        time.sleep(0.5)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        traceback.print_exc()