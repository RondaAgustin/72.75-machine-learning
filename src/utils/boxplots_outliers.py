import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

def main():
    # 1. Asegurar que el directorio de salida exista
    os.makedirs('reports/figures', exist_ok=True)
    
    # 2. Cargar el dataset (con fallback a processed si interim no existe)
    data_path = 'data/interim/train.csv'
    if not os.path.exists(data_path) and os.path.exists('data/processed/train.csv'):
        data_path = 'data/processed/train.csv'
        
    try:
        df = pd.read_csv(data_path, index_col='instant')
        print(f"Dataset cargado exitosamente desde: {data_path}\n")
    except FileNotFoundError:
        print(f"Error: No se encontró el dataset en {data_path}.")
        return
        
    # 3. Definir la lista completa de variables numéricas
    numeric_vars = ['temp', 'atemp', 'hum', 'windspeed']
    
    # Configuración de cuadrícula (3 columnas)
    n_cols = 3
    n_rows = math.ceil(len(numeric_vars) / n_cols)
    
    # 4. Generar cuadrícula de Histogramas
    fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    fig_hist.suptitle('Histogramas de Variables Numéricas', fontsize=16)
    axes_hist_flat = axes_hist.flatten()
    
    for i, col in enumerate(numeric_vars):
        sns.histplot(df[col], kde=True, ax=axes_hist_flat[i], color='skyblue')
        axes_hist_flat[i].set_title(f'Histograma de {col}')
        axes_hist_flat[i].set_xlabel('')
        axes_hist_flat[i].set_ylabel('Frecuencia')
        
    # Ocultar los subplots sobrantes de la cuadrícula
    for j in range(i + 1, len(axes_hist_flat)):
        axes_hist_flat[j].set_visible(False)
        
    plt.tight_layout()
    hist_output = 'reports/figures/histograms.png'
    fig_hist.savefig(hist_output)
    print(f"Figura generada y guardada: {hist_output}")
    plt.close(fig_hist)
    
    # 5. Generar cuadrícula de Boxplots
    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    fig_box.suptitle('Boxplots de Variables Numéricas', fontsize=16)
    axes_box_flat = axes_box.flatten()
    
    for i, col in enumerate(numeric_vars):
        sns.boxplot(x=df[col], ax=axes_box_flat[i], color='lightgreen')
        axes_box_flat[i].set_title(f'Boxplot de {col}')
        axes_box_flat[i].set_xlabel(col)
        
    for j in range(i + 1, len(axes_box_flat)):
        axes_box_flat[j].set_visible(False)
        
    plt.tight_layout()
    box_output = 'reports/figures/boxplots_outliers.png'
    fig_box.savefig(box_output)
    print(f"Figura generada y guardada: {box_output}\n")
    plt.close(fig_box)
    
    # 6. Calcular IQR e imprimir outliers en consola
    print("-" * 50)
    print("REPORTE DE OUTLIERS MATEMÁTICOS (IQR)")
    print("-" * 50)
    
    for col in numeric_vars:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
        print(f"Variable: {col:<10} -> Outliers detectados: {len(outliers)}")
        if len(outliers) > 0:
            outlier_map = outliers[col].to_dict()
            print(f"{col}: {{")
            for idx, val in outlier_map.items():
                print(f"  {idx}: {val}")
            print("}")

if __name__ == '__main__':
    main()