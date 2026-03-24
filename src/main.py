import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from models.linear_regression import run_linear_regression
from models.polynomial_regression import run_polynomial_regression

def load_data():
    train_df = pd.read_csv("../data/processed/train_transformed.csv")
    test_df  = pd.read_csv("../data/processed/test_transformed.csv")

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    return X_train, y_train, X_test, y_test

def main():
    print("Main function is running...")

    X_train, y_train, X_test, y_test = load_data()

    results = []

    # Linear Regression
    lin_model, lin_tr, lin_val = run_linear_regression(X_train, y_train)
    results.append(("Linear", 1, None, lin_model, lin_tr, lin_val))

    # Polynomial (no regularization)
    poly2_model, p2_tr, p2_val = run_polynomial_regression(
        X_train, y_train, degree=2, lambda_l1=0
    )
    results.append(("Poly", 2, 0, poly2_model, p2_tr, p2_val))

    poly3_model, p3_tr, p3_val = run_polynomial_regression(
        X_train, y_train, degree=3, lambda_l1=0
    )
    results.append(("Poly", 3, 0, poly3_model, p3_tr, p3_val))

    # L1 Regularization (Lasso) on polynomial degree 2
    lasso_lambdas = [0.01, 0.1, 1.0]

    for lam in lasso_lambdas:
        model, tr, val = run_polynomial_regression(
            X_train, y_train, degree=2, lambda_l1=lam
        )
        results.append(("Poly+L1", 2, lam, model, tr, val))

    # L1 on degree 3
    for lam in lasso_lambdas:
        model, tr, val = run_polynomial_regression(
            X_train, y_train, degree=3, lambda_l1=lam
        )
        results.append(("Poly+L1", 3, lam, model, tr, val))    

    # Print results
    print("\n=== RESULTS (CV) ===")
    print("Model        Degree   Lambda    Train RMSE    Val RMSE")

    for model_name, deg, lam, _, tr, val in results:
        lam_str = "-" if lam is None else str(lam)
        print(f"{model_name:12} {deg:<8} {lam_str:<9} {tr:12.2f} {val:12.2f}")

    # Find best model (based on validation RMSE)
    best = min(results, key=lambda x: x[5])

    print("\n=== BEST MODEL ===")
    print(f"Model: {best[0]}")
    print(f"Degree: {best[1]}")
    print(f"Lambda: {best[2]}")
    print(f"Validation RMSE: {best[4]:.2f}")

    model = best[3]

    # If polynomial, transform test data
    if "Poly" in best[0]:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=best[1], include_bias=False)
        X_test_transformed = poly.fit_transform(X_test)
    else:
        X_test_transformed = X_test

    y_pred = model.predict(X_test_transformed)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n=== TEST PERFORMANCE ===")
    print(f"Test RMSE: {test_rmse:.2f}")

if __name__ == "__main__":
    main()