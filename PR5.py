#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


def main():
    # Load dataset
    df = pd.read_csv('Social_Network_Ads.csv')

    print("\n--- HEAD ---")
    print(df.head())

    # Encode Gender
    le = LabelEncoder()
    df['Gender_enc'] = le.fit_transform(df['Gender'])

    print("\n--- AFTER ENCODING ---")
    print(df[['Gender', 'Gender_enc']].head())

    # Features and target
    X = df[['Age', 'EstimatedSalary', 'Gender_enc']]
    y = df['Purchased']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("\n--- MODEL TRAINED ---")

    # Predictions
    predictions = model.predict(X_test)

    print("\n--- SAMPLE PREDICTIONS ---")
    for i in range(5):
        print(f"Actual: {y_test.iloc[i]} | Predicted: {predictions[i]}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    print("\n--- CONFUSION MATRIX ---")
    print(cm)

    TN, FP, FN, TP = cm.ravel()

    # Metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    error_rate = (FP + FN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    print("\n--- METRICS ---")
    print("Accuracy:", accuracy)
    print("Error Rate:", error_rate)
    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == "__main__":
    main()