import pandas as pd
import os
import sys

def transform_bike_data(input_path: str, output_path: str) -> None:
    """
    Loads the Bike Sharing dataset, applies data cleaning, 
    feature engineering, and exports the result.
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Remove variables that cause Data Leakage
    # 'casual' and 'registered' sum exactly to the 'cnt' value
    leakage_cols = ['casual', 'registered']
    
    # 2. Remove redundant variables, those without linear predictive value, or identifiers
    # 'temp': highly correlated with 'atemp'
    redundant_cols = ['instant', 'dteday', 'mnth', 'weekday', 'temp', 'yr']
    
    # Execute the removal, checking first if they exist in the DataFrame
    columns_to_drop = [c for c in leakage_cols + redundant_cols if c in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    # 3. Apply One-Hot Encoding (Dummy Variables)
    # We use drop_first=True to avoid the dummy variable trap (multicollinearity)
    categorical_columns = [c for c in ['season', 'weathersit'] if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
    
    # Ensure that the 'weathersit_4' column exists, as the value 4 may not be present in train
    if 'weathersit_4' not in df.columns:
        df['weathersit_4'] = 0
        
    # Rename target variable from 'cnt' to 'target'
    if 'cnt' in df.columns:
        df = df.rename(columns={'cnt': 'target'})
        
    # 4. Export the data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Transformation completed. File successfully saved to: {output_path}")

if __name__ == "__main__":
    # Get filename from arguments or use default
    filename = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_dir, "data", "processed", filename)
    
    # Create output filename dynamically (e.g., train.csv -> train_transformed.csv)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_transformed{ext}"
    output_path = os.path.join(base_dir, "data", "processed", output_filename)

    transform_bike_data(input_path, output_path)