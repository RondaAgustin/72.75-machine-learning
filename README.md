# 72.75-machine-learning

Proyecto de Machine Learning para predecir la cantidad de bicicletas alquiladas por día (`cnt`).

## Requisitos Previos

1. Asegúrate de tener Python instalado.
2. Instala las dependencias del proyecto ejecutando el siguiente comando en la raíz del proyecto:
   ```bash
   pip install -r requirements.txt
   ```

## Instrucciones de Ejecución

Sigue estos pasos para preparar los datos, transformarlos y entrenar los modelos:

### 1. Preparar el Dataset
Este script toma los datos crudos (`data/raw/day.csv`) y los divide en conjuntos de entrenamiento (`train.csv`) y prueba (`test.csv`).
```bash
python src/data/make_dataset.py
```

### 2. Transformar los Datos
Aplica limpieza, eliminación de variables redundantes y One-Hot Encoding. Debes correrlo tanto para el set de entrenamiento como para el de testeo:
```bash
python src/data/transform_data.py train.csv
python src/data/transform_data.py test.csv
```
*(Esto generará los archivos `train_transformed.csv` y `test_transformed.csv` en la carpeta `data/processed/`)*.

### 3. Entrenar y Evaluar los Modelos
Para ejecutar el pipeline principal que entrena los modelos (Regresión Lineal, Polinomial, Lasso) y elige el mejor, **es importante moverte a la carpeta `src`** (ya que el script usa rutas relativas) y luego ejecutar `main.py`:
```bash
cd src
python main.py
```
