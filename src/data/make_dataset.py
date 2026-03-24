
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def make_dataset():
    """
    One-time execution script to load the raw dataset, split it into
    training and test sets, and save them in the processed data folder.
    """
    # --- 1. Path Definitions ---
    # Adjust the base path to work from the project root.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'day.csv')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')

    # --- 2. Load Data ---
    try:
        print(f"Loading raw data from: {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at: {RAW_DATA_PATH}")
        return

    # --- 3. Split Data ---
    print("Splitting data into training and test sets (80/20)...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42  # Seed for reproducibility
    )
    print("Split completed.")
    print(f"Training set size: {train_df.shape}")
    print(f"Test set size: {test_df.shape}")

    # --- 4. Save Processed Data ---
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    train_output_path = os.path.join(PROCESSED_DATA_PATH, 'train.csv')
    test_output_path = os.path.join(PROCESSED_DATA_PATH, 'test.csv')

    print(f"Saving training set to: {train_output_path}")
    train_df.to_csv(train_output_path, index=False)

    print(f"Saving test set to: {test_output_path}")
    test_df.to_csv(test_output_path, index=False)

    print("\nProcess finished. The files train.csv and test.csv have been generated in data/processed/.")

if __name__ == '__main__':
    make_dataset()
