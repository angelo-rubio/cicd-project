
import os
import argparse
import logging
import mlflow
from pathlib import Path


def main():
    """Register the best-trained model in MLflow model registry."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Register trained model in MLflow")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    args = parser.parse_args()

    # Start MLflow logging
    mlflow.start_run()

    try:
        # Log the model path
        logging.info(f"Loading model from: {args.model}")
        print(f"Loading model from: {args.model}")

        # Load the trained model
        model = mlflow.sklearn.load_model(Path(args.model))
        logging.info("Model loaded successfully!")

        # Register the model in MLflow
        logging.info("Registering model in MLflow...")
        print("Registering model in MLflow...")

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_price_regressor",
            registered_model_name="used_cars_price_prediction_model"
        )

        logging.info(f"Model registered successfully! Model info: {model_info}")
        print(f"Model registered successfully!")

    except Exception as e:
        logging.error(f"Error during model registration: {str(e)}", exc_info=True)
        mlflow.end_run()
        raise

    finally:
        # End MLflow run
        mlflow.end_run()


if __name__ == "__main__":
    main()
