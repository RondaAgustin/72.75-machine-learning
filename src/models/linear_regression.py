import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def run_linear_regression(X_train, y_train, k=5):
    print(f"Linear Regression is running...")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    train_rmses = []
    val_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        model = LinearRegression()
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])

        train_rmse = np.sqrt(mean_squared_error(y_train.iloc[train_idx], model.predict(X_train.iloc[train_idx])))
        val_rmse   = np.sqrt(mean_squared_error(y_train.iloc[val_idx],   model.predict(X_train.iloc[val_idx])))

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        print(f"  Fold {fold}: train={train_rmse:.2f}  val={val_rmse:.2f}")

    mean_train = np.mean(train_rmses)
    mean_val   = np.mean(val_rmses)
    print(f"  Mean:   train={mean_train:.2f}  val={mean_val:.2f}\n")

    # Retrain on the full training set
    final_model = LinearRegression()
    final_model.fit(X_train, y_train)
    return final_model, mean_train, mean_val