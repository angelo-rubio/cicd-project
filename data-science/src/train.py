import argparse
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os


def main():
    """Train a Random Forest Regressor model on used cars data."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Random Forest model for car price prediction")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data folder")
    parser.add_argument("--test_data", type=str, required=True, help="Path to testing data folder")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of trees")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save the trained model")

    args = parser.parse_args()

    # Handle max_depth (argparse doesn't handle None well)
    if args.max_depth == -1 or args.max_depth is None:
        args.max_depth = None

    # Start MLflow logging
    mlflow.start_run()

    try:
        # Read the training and testing data
        logging.info(f"Reading training data from: {args.train_data}")
        train_df = pd.read_csv(os.path.join(args.train_data, "train.csv"))
        logging.info(f"Training data shape: {train_df.shape}")

        logging.info(f"Reading testing data from: {args.test_data}")
        test_df = pd.read_csv(os.path.join(args.test_data, "test.csv"))
        logging.info(f"Testing data shape: {test_df.shape}")

        # Separate features (X) and target (y)
        target_column = 'price'

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Convert categorical columns to numeric using one-hot encoding
        logging.info("Converting categorical text to numbers...")
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)

        # Ensure both train and test have the exact same columns after encoding
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        logging.info(f"Training set size: {len(X_train)}")
        logging.info(f"Testing set size: {len(X_test)}")
        logging.info(f"Number of features after encoding: {X_train.shape[1]}")

        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        print(f"Number of features after encoding: {X_train.shape[1]}")

        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Initialize and train the Random Forest Regressor
        logging.info("Training Random Forest Regressor...")
        print("Training Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)
        logging.info("Model training completed!")

        # Evaluate the model
        logging.info("Evaluating model...")
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        logging.info(f"Mean Squared Error: {mse}")
        print(f"Mean Squared Error: {mse}")

        # Log the MSE metric
        mlflow.log_metric("MSE", mse)

        # Save the model
        logging.info(f"Saving model to: {args.model_output}")
        print(f"Saving model to: {args.model_output}")
        mlflow.sklearn.save_model(model, args.model_output)

        logging.info("Model training completed successfully!")
        print("Model training completed successfully!")

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}", exc_info=True)
        mlflow.end_run()
        raise

    finally:
        # End MLflow run
        mlflow.end_run()


if __name__ == "__main__":
    main()
