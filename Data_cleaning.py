import pandas as pd
import numpy as np

def lines():
    '''Creates Line breaks for improved readability'''
    print('-' * 150)


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('Customer-Churn-Data.csv')

    # Read the data information
    df.info()
    lines()

    # Find Missing Values
    df_missing = df[df.isna().any(axis=1)]
    print(df_missing)
    lines()

    # View duplicates
    df_duplicates = df[df.duplicated(keep=False)]
    print(df_duplicates)
    lines()

    # Check the data shape
    observations , features = df.shape
    print(f'The dataset has {observations} observations and {features} features')
    lines()

    # Convert data with numbers stored as text
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') 

    # Confirm Conversion
    df.info()
    lines()

    # Recheck for missing values
    df_missing = df[df.isna().any(axis=1)]
    print(df_missing)
    lines()

    avg_total_charges = df['TotalCharges'].mean()
    df['TotalCharges'] = df['TotalCharges'].fillna(round(avg_total_charges, 2))
    df.info()
    lines()

    # Drop Customer ID
    df = df.drop(columns=["customerID"])

    # Export Cleaned Data
    df.to_csv('Customer-Churn-Data-cleaned.csv', index=False)
    print('Cleaned Data Exported!')