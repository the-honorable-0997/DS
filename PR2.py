#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


def main():
    # Create dataset
    data = {
        'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'Math': [85, 90, np.nan, 75, 65, 200, 70],
        'English': [88, 92, 95, 67, np.nan, 63, 77],
        'Science': [78, 85, 82, 60, 90, 88, np.nan]
    }

    df = pd.DataFrame(data)

    print("\n--- ORIGINAL DATA ---")
    print(df)

    print("\n--- NULL VALUES ---")
    print(df.isnull().sum())

    # Fill missing values
    df['Math'] = df['Math'].fillna(df['Math'].mean())
    df['English'].fillna(df['English'].median(), inplace=True)
    df['Science'] = df['Science'].fillna(df['Science'].median())

    print("\n--- AFTER FILLING NULLS ---")
    print(df)

    # Cap extreme values (basic cleaning)
    df['Math'] = df['Math'].apply(lambda x: 100 if x > 100 else x)

    print("\n--- AFTER CAPPING (>100) ---")
    print(df)

    # IQR Outlier Handling
    Q1 = df['Math'].quantile(0.25)
    Q3 = df['Math'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print("\n--- IQR RANGE ---")
    print("Lower:", lower, "Upper:", upper)

    # Cap outliers
    df['Math'] = np.where(df['Math'] > upper, upper, df['Math'])

    print("\n--- AFTER IQR OUTLIER HANDLING ---")
    print(df)

    # Normalization and log transform
    df['Math_normalized'] = (df['Math'] - df['Math'].min()) / (df['Math'].max() - df['Math'].min())
    df['Math_log'] = np.log(df['Math'])

    print("\n--- FINAL DATA ---")
    print(df)


if __name__ == "__main__":
    main()