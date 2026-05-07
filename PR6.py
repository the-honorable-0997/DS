#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


def main():
    # Load dataset
    df = pd.read_csv("Iris.csv")

    print("\n--- HEAD ---")
    print(df.head())

    # Features & target
    X = df.drop('Species', axis=1)
    y = df['Species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)

    print("\n--- MODEL TRAINED ---")

    # Predictions
    y_pred = model.predict(X_test)

    print("\n--- SAMPLE PREDICTIONS ---")
    for i in range(5):
        print(f"Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]}")

    # Confusion matrix (multiclass)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- CONFUSION MATRIX (MULTI-CLASS) ---")
    print(cm)

    # -------------------------------
    # Binary: Setosa vs Rest
    # -------------------------------
    y_test_bin = (y_test == 'Iris-setosa')
    y_pred_bin = (y_pred == 'Iris-setosa')

    cm2 = confusion_matrix(y_test_bin, y_pred_bin)

    print("\n--- CONFUSION MATRIX (SETOSA vs REST) ---")
    print(cm2)

    TN, FP, FN, TP = cm2.ravel()

    print("\n--- FOR CLASS: SETOSA ---")
    print("TP:", TP)
    print("FP:", FP)
    print("TN:", TN)
    print("FN:", FN)

    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = (FP + FN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    print("\n--- METRICS ---")
    print("Accuracy:", accuracy)
    print("Error Rate:", error_rate)
    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == "__main__":
    main()