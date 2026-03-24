
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the dataset
try:
    df = pd.read_csv('data/processed/train.csv')
except FileNotFoundError:
    print("Error: The file data/processed/train.csv was not found.")
    print("Please make sure the path is correct.")
    exit()


# Define categorical variables and the target
categorical_vars = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
target = 'cnt'

# Ensure the output directory exists
os.makedirs('reports/figures', exist_ok=True)

# --- Categorical variables countplots ---
plt.figure(figsize=(20, 15))
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=var, data=df)
    plt.title(f'{var} Frequency')
plt.tight_layout()
plt.savefig('reports/figures/categorical_frequencies.png')
plt.close()

print("reports/figures/categorical_frequencies.png has been saved.")

# --- Categorical variables vs. target boxplots ---
plt.figure(figsize=(20, 15))
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=var, y=target, data=df)
    plt.title(f'{var} vs. {target}')
plt.tight_layout()
plt.savefig('reports/figures/boxplots_cnt.png')
plt.close()

print("reports/figures/boxplots_cnt.png ha sido guardado.")
