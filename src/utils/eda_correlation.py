import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    # Asegurar que el directorio de salida exista
    os.makedirs('reports/figures', exist_ok=True)
    
    # Cargar el dataset limpio
    data_path = 'data/processed/train.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el dataset en {data_path}.")
        return

    # Seleccionar únicamente las variables numéricas
    numeric_vars = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    df_numeric = df[numeric_vars]

    # Calcular la Pearson Correlation Matrix usando Pandas
    corr_matrix = df_numeric.corr(method='pearson')

    # Generar figura visualizando esta matriz con un Heatmap de Seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0, 
        fmt=".2f"
    )
    plt.title('Pearson Correlation Matrix', fontsize=16)
    
    # Guardar la figura
    output_path = 'reports/figures/correlation_matrix.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    # Imprimir en consola el reporte de alta correlación
    print("High Correlation Pairs (>0.75):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.75:
                print(f" - {var1} & {var2}: {corr_val:.4f}")

if __name__ == "__main__":
    main()
