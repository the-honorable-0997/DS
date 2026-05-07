#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def main():
    # Load dataset
    df = pd.read_csv("Titanic-Dataset.csv")

    print("\n--- HEAD ---")
    print(df.head())

    print("\n--- DESCRIPTION ---")
    print(df.describe())

    print("\n--- INFO ---")
    print(df.info())

    # Convert Age to float
    df['Age'] = df['Age'].astype(float)

    print("\n--- SHAPE ---")
    print(df.shape)

    print("\n--- NULL VALUES ---")
    print(df.isnull().sum())

    # Convert Sex to string
    df['Sex'] = df['Sex'].astype('string')

    print("\n--- INFO AFTER TYPE CHANGE ---")
    print(df.info())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    # Normalization (Min-Max)
    df['Age_norm2'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())

    print("\n--- NORMALIZED AGE ---")
    print(df[['Age_norm2']].head())

    # One-hot encoding
    df2 = pd.get_dummies(df, columns=['Embarked'])

    print("\n--- ONE HOT ENCODED (Embarked) ---")
    print(df2.head())

    # Label Encoding
    le = LabelEncoder()
    df['Embarked_enc'] = le.fit_transform(df['Embarked'].astype(str))

    print("\n--- LABEL ENCODED ---")
    print(df[['Embarked', 'Embarked_enc']].head())


if __name__ == "__main__":
    main()