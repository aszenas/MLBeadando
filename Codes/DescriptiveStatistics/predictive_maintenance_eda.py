
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# Adatok betöltése
df = pd.read_csv("predictive_maintenance.csv")

# Adatkészlet áttekintése
print("Adatkészlet mérete:", df.shape)
print(df.info())

# Leíró statisztikák
print(df.describe())

# Hiányzó értékek
print(df.isnull().sum())

# Célváltozó elemzése
plt.figure(figsize=(6,4))
sns.countplot(x='Target', data=df)
plt.title('A célváltozó eloszlása')
plt.show()

print(df['Target'].value_counts())
print(df['Target'].value_counts(normalize=True) * 100)

# Kategorikus változók
plt.figure(figsize=(6,4))
sns.countplot(x='Type', data=df)
plt.title('Terméktípusok eloszlása')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(y='Failure Type', data=df)
plt.title('Hibatípusok eloszlása')
plt.show()

# Numerikus változók
numeric_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} eloszlása')
    plt.show()

# Outlier vizsgálat
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'{col} boxplot')
    plt.show()

# Korreláció
plt.figure(figsize=(10,8))

corr = df.select_dtypes(include=np.number).corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelációs mátrix')
plt.show()

# Kapcsolat a célváltozóval
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Target', y=col, data=df)
    plt.title(f'{col} és a Target kapcsolata')
    plt.show()

# Adatelőkészítés
df_model = df.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)

df_model = pd.get_dummies(df_model, columns=['Type'], drop_first=True)

print(df_model.head())
