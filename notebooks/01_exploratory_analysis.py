# 01_exploratory_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path="data/raw/churn_data.csv"):
    df = pd.read_csv(data_path)
    print(df.head())
    print(df.describe())
    print(df.info())
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='churn')
    plt.title('Churn Distribution')
    plt.savefig('churn_distribution.png')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')

if __name__ == "__main__":
    # run_eda()
    pass
