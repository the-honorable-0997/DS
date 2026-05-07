#!/usr/bin/env python
# coding: utf-8

import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Load dataset
    df = sns.load_dataset('titanic')

    print("\n--- HEAD ---")
    print(df.head())

    print("\n--- DESCRIPTION ---")
    print(df.describe())

    print("\n--- INFO ---")
    print(df.info())

    # Count plot (gender)
    plt.figure()
    sns.countplot(x='sex', data=df)
    plt.title("Gender Count")
    plt.show()

    # Count plot (gender vs survival)
    plt.figure()
    sns.countplot(x='sex', hue='survived', data=df)
    plt.title("Gender vs Survival")
    plt.show()

    # Count plot (class vs survival)
    plt.figure()
    sns.countplot(x='pclass', hue='survived', data=df)
    plt.title("Passenger Class vs Survival")
    plt.show()

    # Histogram (fare distribution)
    plt.figure()
    plt.hist(df['fare'], bins=20)
    plt.xlabel("Fare")
    plt.ylabel("Number of Passengers")
    plt.title("Fare Distribution")
    plt.show()


if __name__ == "__main__":
    main()