import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/student_performance.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "student_performance_model.pkl"
)

RESULTS_DIR = "outputs/predictions"
RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "model_results.csv"
)

TARGET_COLUMN = "final_score"


# Create required directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_data():
    """
    Load the student performance dataset.
    """

    try:
        df = pd.read_csv(DATA_PATH)

        print("Dataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")

        return df

    except FileNotFoundError:

        print(
            f"ERROR: Dataset not found at: {DATA_PATH}"
        )

        return None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """
    Clean dataset and separate features and target.
    """

    # Remove duplicate rows
    df = df.drop_duplicates().copy()

    # Check target column
    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in the dataset."
        )

    # Separate features and target
    X = df.drop(
        columns=[
            TARGET_COLUMN,
            "student_id"
        ]
    )

    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Identify numerical and categorical features
    # --------------------------------------------------------

    # Explicitly select numerical columns
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Explicitly select categorical columns
    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    return (
        X,
        y,
        numerical_features,
        categorical_features
    )


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Create preprocessing pipeline for numerical
    and categorical features.
    """

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Calculate regression evaluation metrics.
    """

    # Generate predictions
    predictions = model.predict(X_test)

    # Mean Absolute Error
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # Mean Squared Error
    mse = mean_squared_error(
        y_test,
        predictions
    )

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # R² Score
    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return metrics


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    """
    Train multiple regression models and evaluate them.
    """

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=42
            )
    }

    trained_models = {}
    results = {}

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for model_name, regressor in models.items():

        print("\n")
        print("=" * 60)
        print(f"Training Model: {model_name}")
        print("=" * 60)

        # Create complete ML pipeline
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "regressor",
                    regressor
                )
            ]
        )

        # Train model
        pipeline.fit(
            X_train,
            y_train
        )

        # Evaluate model
        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        # Store model
        trained_models[model_name] = pipeline

        # Store results
        results[model_name] = metrics

        # Display results
        print(
            f"MAE  : {metrics['MAE']:.4f}"
        )

        print(
            f"MSE  : {metrics['MSE']:.4f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.4f}"
        )

        print(
            f"R²   : {metrics['R2']:.4f}"
        )

    return trained_models, results


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    trained_models,
    results
):
    """
    Select the model with the highest R² score.
    """

    # Find model with highest R²
    best_model_name = max(
        results,
        key=lambda model:
        results[model]["R2"]
    )

    best_model = trained_models[
        best_model_name
    ]

    print("\n")
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Selected Model: {best_model_name}"
    )

    print(
        f"R² Score: "
        f"{results[best_model_name]['R2']:.4f}"
    )

    print(
        f"MAE: "
        f"{results[best_model_name]['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{results[best_model_name]['RMSE']:.4f}"
    )

    return (
        best_model_name,
        best_model
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """
    Save the trained model to a pickle file.
    """

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    print("\n")
    print(
        "Model saved successfully:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

def save_results(results):
    """
    Save model evaluation results as CSV.
    """

    rows = []

    for model_name, metrics in results.items():

        rows.append({

            "Model": model_name,

            "MAE": round(
                metrics["MAE"],
                4
            ),

            "MSE": round(
                metrics["MSE"],
                4
            ),

            "RMSE": round(
                metrics["RMSE"],
                4
            ),

            "R2": round(
                metrics["R2"],
                4
            )
        })

    results_df = pd.DataFrame(
        rows
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        "\nModel comparison results saved:"
    )

    print(
        RESULTS_PATH
    )

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION")
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load Dataset
    # --------------------------------------------------------

    df = load_data()

    if df is None:
        return

    # --------------------------------------------------------
    # Step 2: Prepare Data
    # --------------------------------------------------------

    (
        X,
        y,
        numerical_features,
        categorical_features
    ) = prepare_data(df)

    print("\n")
    print("Numerical Features:")
    print(numerical_features)

    print("\n")
    print("Categorical Features:")
    print(categorical_features)

    print("\n")
    print("Target Variable:")
    print(TARGET_COLUMN)

    # --------------------------------------------------------
    # Step 3: Split Dataset
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )

    print("\n")
    print(
        f"Training Samples: {len(X_train)}"
    )

    print(
        f"Testing Samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Step 4: Create Preprocessing Pipeline
    # --------------------------------------------------------

    preprocessor = create_preprocessor(

        numerical_features,

        categorical_features
    )

    # --------------------------------------------------------
    # Step 5: Train Models
    # --------------------------------------------------------

    (
        trained_models,
        results
    ) = train_models(

        X_train,

        X_test,

        y_train,

        y_test,

        preprocessor
    )

    # --------------------------------------------------------
    # Step 6: Select Best Model
    # --------------------------------------------------------

    (
        best_model_name,
        best_model
    ) = select_best_model(

        trained_models,

        results
    )

    # --------------------------------------------------------
    # Step 7: Save Best Model
    # --------------------------------------------------------

    save_model(
        best_model
    )

    # --------------------------------------------------------
    # Step 8: Save Evaluation Results
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nFinal Selected Model: "
        f"{best_model_name}"
    )

    print(
        f"Final R² Score: "
        f"{results[best_model_name]['R2']:.4f}"
    )

    print(
        "\nYour trained model is ready "
        "for student performance prediction."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()