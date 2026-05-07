#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # Load dataset
    df = pd.read_csv('housing.csv')

    print("\n--- HEAD ---")
    print(df.head())

    # Split features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n--- DATA SPLIT ---")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n--- MODEL TRAINED ---")

    # Predictions
    predictions = model.predict(X_test)

    print("\n--- SAMPLE PREDICTIONS ---")
    for i in range(5):
        print(f"Actual: {y_test.iloc[i]:.2f} | Predicted: {predictions[i]:.2f}")

    # Evaluation
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- MODEL PERFORMANCE ---")
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)


if __name__ == "__main__":
    main()