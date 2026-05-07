#!/usr/bin/env python
# coding: utf-8

import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Load dataset
    df = sns.load_dataset('titanic')

    print("\n--- HEAD ---")
    print(df.head())

    print("\n--- INFO ---")
    print(df.info())

    # Boxplot
    plt.figure()
    sns.boxplot(x='sex', y='age', hue='survived', data=df)

    plt.title("Age Distribution by Gender and Survival")
    plt.xlabel("Gender")
    plt.ylabel("Age")

    plt.show()


if __name__ == "__main__":
    main()