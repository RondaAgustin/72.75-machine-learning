import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def run_polynomial_regression(X_train, y_train, degree=2, lambda_l1=None, k=5):
    print(f"Polynomial Regression degree={degree} and lamda={lambda_l1} is running...")
    
    # Apply polynomial transformation
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    train_rmses = []
    val_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_poly), start=1):        
        # Train on transformed features / Use Lasso if lambda provided
        model = Lasso(alpha=lambda_l1, max_iter=100000) if lambda_l1 else LinearRegression()
        model.fit(X_poly[train_idx], y_train.iloc[train_idx])

        train_rmse = np.sqrt(mean_squared_error(y_train.iloc[train_idx], model.predict(X_poly[train_idx])))
        val_rmse   = np.sqrt(mean_squared_error(y_train.iloc[val_idx],   model.predict(X_poly[val_idx])))

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        # print(f"  Fold {fold}: train={train_rmse:.2f}  val={val_rmse:.2f}")

    mean_train = np.mean(train_rmses)
    mean_val   = np.mean(val_rmses)
    print(f"  Mean:   train={mean_train:.2f}  val={mean_val:.2f}\n")

    # Retrain on the full training set
    final_model = Lasso(alpha=lambda_l1, max_iter=10000) if lambda_l1 else LinearRegression()
    final_model.fit(X_poly, y_train)
    return final_model, mean_train, mean_val