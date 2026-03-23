
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def make_dataset():
    """
    Script de ejecución única para cargar el dataset crudo, dividirlo en conjuntos
    de entrenamiento y prueba, y guardarlos en la carpeta de datos procesados.
    """
    # --- 1. Definición de Rutas ---
    # Ajusta la ruta base para que funcione desde la raíz del proyecto.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'day.csv')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')

    # --- 2. Carga de Datos ---
    try:
        print(f"Cargando datos crudos desde: {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        print("Datos crudos cargados exitosamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en: {RAW_DATA_PATH}")
        return

    # --- 3. División de Datos ---
    print("Dividiendo los datos en conjuntos de entrenamiento y prueba (80/20)...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42  # Semilla para reproducibilidad
    )
    print("División completada.")
    print(f"Tamaño del conjunto de entrenamiento: {train_df.shape}")
    print(f"Tamaño del conjunto de prueba: {test_df.shape}")

    # --- 4. Guardado de Datos Procesados ---
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    train_output_path = os.path.join(PROCESSED_DATA_PATH, 'train.csv')
    test_output_path = os.path.join(PROCESSED_DATA_PATH, 'test.csv')

    print(f"Guardando conjunto de entrenamiento en: {train_output_path}")
    train_df.to_csv(train_output_path, index=False)

    print(f"Guardando conjunto de prueba en: {test_output_path}")
    test_df.to_csv(test_output_path, index=False)

    print("\nProceso finalizado. Los archivos train.csv y test.csv se han generado en data/processed/.")

if __name__ == '__main__':
    make_dataset()
