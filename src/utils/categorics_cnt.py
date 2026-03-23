
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Cargar el dataset
try:
    df = pd.read_csv('data/processed/train.csv')
except FileNotFoundError:
    print("Error: El archivo data/processed/train.csv no fue encontrado.")
    print("Por favor, asegúrese que el path es el correcto.")
    exit()


# Definir variables categóricas y el target
categorical_vars = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
target = 'cnt'

# Asegurar que el directorio de salida exista
os.makedirs('reports/figures', exist_ok=True)

# --- Countplots de variables categóricas ---
plt.figure(figsize=(20, 15))
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=var, data=df)
    plt.title(f'Frecuencia de {var}')
plt.tight_layout()
plt.savefig('reports/figures/frecuencias_categoricas.png')
plt.close()

print("reports/figures/frecuencias_categoricas.png ha sido guardado.")

# --- Boxplots de variables categóricas vs. target ---
plt.figure(figsize=(20, 15))
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=var, y=target, data=df)
    plt.title(f'{var} vs. {target}')
plt.tight_layout()
plt.savefig('reports/figures/boxplots_cnt.png')
plt.close()

print("reports/figures/boxplots_cnt.png ha sido guardado.")
