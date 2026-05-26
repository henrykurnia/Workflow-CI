import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


# aktifkan autolog
mlflow.autolog()


# load dataset preprocessing
df = pd.read_csv("Retail_Transaction_Dataset_Preprocessing.csv")


# pisahkan fitur dan target
X = df.drop("TotalAmount", axis=1)
y = df["TotalAmount"]


# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# training dengan MLflow
with mlflow.start_run(run_name="Retail_RF_Model"):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)

    # ==========================================
    # 1. FEATURE IMPORTANCE
    # ==========================================

    importance = model.feature_importances_

    feature_names = X.columns

    plt.figure(figsize=(10,6))

    plt.barh(feature_names, importance)

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("feature_importance.png")

    mlflow.log_artifact("feature_importance.png")

    plt.close()

    # ==========================================
    # 2. ACTUAL VS PREDICTED
    # ==========================================

    plt.figure(figsize=(7,7))

    plt.scatter(y_test, y_pred)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.title("Actual vs Predicted")

    plt.tight_layout()

    plt.savefig("actual_vs_predicted.png")

    mlflow.log_artifact("actual_vs_predicted.png")

    plt.close()

    # ==========================================
    # 3. RESIDUAL PLOT
    # ==========================================

    residuals = y_test - y_pred

    plt.figure(figsize=(7,7))

    plt.scatter(y_pred, residuals)

    plt.axhline(y=0)

    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig("residual_plot.png")

    mlflow.log_artifact("residual_plot.png")

    plt.close()