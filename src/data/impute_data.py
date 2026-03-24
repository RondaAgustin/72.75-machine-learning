import pandas as pd
from typing import List, Any
import os
import sys

def impute_humidity_by_instant(file_path: str, target_instant: int) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame and imputes an anomalous humidity ('hum') 
    value for a specific row identified by 'instant'. The imputation is based on 
    the mean humidity of similar instances (neighbors) based on 'atemp', 'windspeed', 
    and 'weathersit'. 
    
    The function does not modify the original CSV file on disk.
    
    Args:
        file_path (str): The path to the CSV file.
        target_instant (int): The 'instant' ID of the row with the anomalous humidity.
        
    Returns:
        pd.DataFrame: A modified DataFrame with the imputed 'hum' value.
    """
    
    # 1. Load data
    df = pd.read_csv(file_path)
    
    # Check if the target_instant exists in the DataFrame
    if target_instant not in df['instant'].values:
        print(f"Error: Target instant '{target_instant}' not found in the dataset.")
        return df
    
    # 2. Identify the target row
    target_mask = df['instant'] == target_instant
    target_row = df[target_mask].iloc[0]
    
    target_atemp = target_row['atemp']
    target_windspeed = target_row['windspeed']
    target_weathersit = target_row['weathersit']
    
    # Define a mask to exclude the target row from the neighbors search
    others_mask = df['instant'] != target_instant
    
    # 3. Search for strict neighbors
    # Condition: +/- 0.05 for atemp, +/- 0.05 for windspeed, and exact match for weathersit
    strict_mask = others_mask & \
                  (df['atemp'] >= target_atemp - 0.05) & \
                  (df['atemp'] <= target_atemp + 0.05) & \
                  (df['windspeed'] >= target_windspeed - 0.05) & \
                  (df['windspeed'] <= target_windspeed + 0.05) & \
                  (df['weathersit'] == target_weathersit)
                  
    strict_neighbors = df[strict_mask]
    
    # Initialize variables to hold imputation results
    mean_hum: float = 0.0
    used_instants: List[Any] = []
    
    # 4. Impute (Attempt 1: Strict)
    if not strict_neighbors.empty:
        mean_hum = strict_neighbors['hum'].mean()
        used_instants = strict_neighbors['instant'].tolist()
        
    else:
        # 5. Relax the search (Attempt 2: Relaxed)
        # Drop the exact match condition for 'weathersit'
        relaxed_mask = others_mask & \
                       (df['atemp'] >= target_atemp - 0.05) & \
                       (df['atemp'] <= target_atemp + 0.05) & \
                       (df['windspeed'] >= target_windspeed - 0.05) & \
                       (df['windspeed'] <= target_windspeed + 0.05)
                       
        relaxed_neighbors = df[relaxed_mask]
        
        if not relaxed_neighbors.empty:
            mean_hum = relaxed_neighbors['hum'].mean()
            used_instants = relaxed_neighbors['instant'].tolist()
        else:
            # Failsafe in case no neighbors are found even with relaxed rules
            print("No neighbors found with the given criteria. Returning original DataFrame.")
            return df
            
    # 6. Replace and Report
    # Update the target row's humidity with the calculated mean
    df.loc[target_mask, 'hum'] = mean_hum
    
    print(f"Target instant: {target_instant}")
    print(f"New imputed humidity: {mean_hum}")
    print(f"Used neighbor instants for calculation: {used_instants}")
    
    # 7. Return modified DataFrame
    return df

if __name__ == "__main__":
    # Get filename from arguments or use default
    filename = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    
    # Determine the project's base path to be able to run the script from any directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_dir, "data", "processed", filename)
    
    # The anomalous humidity record discovered is instant 69
    target = 69
    
    print(f"Starting humidity imputation on {filename}...")
    corrected_df = impute_humidity_by_instant(input_path, target)