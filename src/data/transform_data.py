import pandas as pd
import os

def transform_bike_data(input_path: str, output_path: str) -> None:
    """
    Carga el dataset de Bike Sharing, aplica limpieza de datos, 
    ingeniería de características y exporta el resultado.
    """
    print(f"Cargando datos desde: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Eliminar variables que generan Data Leakage
    # 'casual' y 'registered' suman exactamente el valor de 'cnt'
    cols_leakage = ['casual', 'registered']
    
    # 2. Eliminar variables redundantes, sin valor predictivo lineal o identificadores
    # 'temp': altamente correlacionada con 'atemp'
    cols_redundantes = ['instant', 'dteday', 'mnth', 'weekday', 'temp', 'yr']
    
    # Ejecutamos la eliminación comprobando primero que existan en el DataFrame
    columnas_a_eliminar = [c for c in cols_leakage + cols_redundantes if c in df.columns]
    df = df.drop(columns=columnas_a_eliminar)
    
    # 3. Aplicar One-Hot Encoding (Variables Dummy)
    # Utilizamos drop_first=True para evitar la trampa de variables dummy (multicolinealidad)
    columnas_categoricas = [c for c in ['season', 'weathersit'] if c in df.columns]
    df = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True, dtype=int)
    
    # Garantizar que la columna 'weathersit_4' exista, ya que el valor 4 puede no estar presente en train
    if 'weathersit_4' not in df.columns:
        df['weathersit_4'] = 0
        
    # 4. Exportar los datos
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Transformación completada. Archivo guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ruta_entrada = os.path.join(base_dir, "data", "processed", "train.csv")
    ruta_salida = os.path.join(base_dir, "data", "processed", "train_transformed.csv")
    
    transform_bike_data(ruta_entrada, ruta_salida)