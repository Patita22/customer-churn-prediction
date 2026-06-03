import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.head())

print(df.info())

# Convert TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(0, inplace=True)

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Boxplots
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in numeric_cols:
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(col)
    plt.show()

# Heatmap
corr = df[['tenure','MonthlyCharges','TotalCharges','Churn']].corr()

sns.heatmap(corr, annot=True)

plt.show()

# Encoding
cat_cols = df.select_dtypes(include='object').columns.tolist()

cat_cols.append('SeniorCitizen')

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print(df_encoded.head())