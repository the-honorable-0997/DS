#!/usr/bin/env python
# coding: utf-8

import pandas as pd


def main():
    # Load dataset
    df = pd.read_csv('Iris.csv')

    print("\n--- HEAD ---")
    print(df.head())

    # Grouping
    grouped = df.groupby('Species')

    print("\n--- GROUPED DESCRIPTION (SepalLengthCm) ---")
    print(grouped['SepalLengthCm'].describe())

    # Extract lists
    setosa_list = df[df['Species'] == 'Iris-setosa']['SepalLengthCm'].tolist()
    versicolor_list = df[df['Species'] == 'Iris-versicolor']['SepalLengthCm'].tolist()
    virginica_list = df[df['Species'] == 'Iris-virginica']['SepalLengthCm'].tolist()

    print("\n--- SEPAL LENGTH LISTS ---")
    print("Setosa:", setosa_list)
    print("Versicolor:", versicolor_list)
    print("Virginica:", virginica_list)

    # Create separate DataFrames
    setosa = df[df['Species'] == 'Iris-setosa']
    versicolor = df[df['Species'] == 'Iris-versicolor']
    virginica = df[df['Species'] == 'Iris-virginica']

    print("\n--- DESCRIPTIVE STATS PER SPECIES ---")
    print("\nSetosa:\n", setosa.describe())
    print("\nVersicolor:\n", versicolor.describe())
    print("\nVirginica:\n", virginica.describe())


if __name__ == "__main__":
    main()