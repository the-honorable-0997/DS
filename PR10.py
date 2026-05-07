#!/usr/bin/env python
# coding: utf-8

import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Load dataset
    df = sns.load_dataset('iris')

    print("\n--- HEAD ---")
    print(df.head())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    # Histograms
    print("\n--- HISTOGRAMS ---")
    df.hist(figsize=(8, 6))
    plt.tight_layout()
    plt.show()

    # Boxplots for each feature
    print("\n--- BOXPLOTS ---")
    for col in df.columns[:-1]:  # exclude 'species'
        plt.figure()
        sns.boxplot(y=df[col])
        plt.title(col)
        plt.show()

    # Outliers for sepal_width (detailed)
    Q1 = df['sepal_width'].quantile(0.25)
    Q3 = df['sepal_width'].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[
        (df['sepal_width'] < Q1 - 1.5 * IQR) |
        (df['sepal_width'] > Q3 + 1.5 * IQR)
    ]

    print("\n--- OUTLIERS (sepal_width) ---")
    print(outliers)

    # Outlier count for all features
    print("\n--- OUTLIER COUNT PER FEATURE ---")
    for feature in df.columns[:-1]:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[feature] < lower) | (df[feature] > upper)]
        print(f"{feature}: {len(outliers)} outliers found")


if __name__ == "__main__":
    main()