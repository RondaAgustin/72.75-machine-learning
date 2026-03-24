import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

def main():
    # 1. Ensure the output directory exists
    os.makedirs('reports/figures', exist_ok=True)
    
    # 2. Load the dataset (with fallback to processed if interim doesn't exist)
    data_path = 'data/interim/train.csv'
    if not os.path.exists(data_path) and os.path.exists('data/processed/train.csv'):
        data_path = 'data/processed/train.csv'
        
    try:
        df = pd.read_csv(data_path, index_col='instant')
        print(f"Dataset loaded successfully from: {data_path}\n")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}.")
        return
        
    # 3. Define the full list of numeric variables
    numeric_vars = ['temp', 'atemp', 'hum', 'windspeed']
    
    # Grid configuration (3 columns)
    n_cols = 3
    n_rows = math.ceil(len(numeric_vars) / n_cols)
    
    # 4. Generate Histograms grid
    fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    fig_hist.suptitle('Numeric Variables Histograms', fontsize=16)
    axes_hist_flat = axes_hist.flatten()
    
    for i, col in enumerate(numeric_vars):
        sns.histplot(df[col], kde=True, ax=axes_hist_flat[i], color='skyblue')
        axes_hist_flat[i].set_title(f'{col} Histogram')
        axes_hist_flat[i].set_xlabel('')
        axes_hist_flat[i].set_ylabel('Frequency')
        
    # Hide any extra subplots in the grid
    for j in range(i + 1, len(axes_hist_flat)):
        axes_hist_flat[j].set_visible(False)
        
    plt.tight_layout()
    hist_output = 'reports/figures/histograms.png'
    fig_hist.savefig(hist_output)
    print(f"Figure generated and saved: {hist_output}")
    plt.close(fig_hist)
    
    # 5. Generate Boxplots grid
    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    fig_box.suptitle('Numeric Variables Boxplots', fontsize=16)
    axes_box_flat = axes_box.flatten()
    
    for i, col in enumerate(numeric_vars):
        sns.boxplot(x=df[col], ax=axes_box_flat[i], color='lightgreen')
        axes_box_flat[i].set_title(f'{col} Boxplot')
        axes_box_flat[i].set_xlabel(col)
        
    for j in range(i + 1, len(axes_box_flat)):
        axes_box_flat[j].set_visible(False)
        
    plt.tight_layout()
    box_output = 'reports/figures/boxplots_outliers.png'
    fig_box.savefig(box_output)
    print(f"Figure generated and saved: {box_output}\n")
    plt.close(fig_box)
    
    # 6. Calculate IQR and print outliers to console
    print("-" * 50)
    print("MATHEMATICAL OUTLIERS REPORT (IQR)")
    print("-" * 50)
    
    for col in numeric_vars:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Variable: {col:<10} -> Detected outliers: {len(outliers)}")
        if len(outliers) > 0:
            outlier_map = outliers[col].to_dict()
            print(f"{col}: {{")
            for idx, val in outlier_map.items():
                print(f"  {idx}: {val}")
            print("}")

if __name__ == '__main__':
    main()