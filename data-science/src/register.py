
import os
import argparse
import logging
import mlflow
from pathlib import Path
import json


def main():
    """Register the best-trained model in MLflow model registry."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Register trained model in MLflow")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the registered model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--model_info_output_path", type=str, required=True, help="Path to save model info")
    args = parser.parse_args()

    # Start MLflow logging
    mlflow.start_run()

    try:
        # Log the model path
        logging.info(f"Loading model from: {args.model_path}")
        print(f"Loading model from: {args.model_path}")

        # Load the trained model
        model = mlflow.sklearn.load_model(Path(args.model_path))
        logging.info("Model loaded successfully!")

        # Register the model in MLflow
        logging.info(f"Registering model '{args.model_name}' in MLflow...")
        print(f"Registering model '{args.model_name}' in MLflow...")

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_price_regressor",
            registered_model_name=args.model_name
        )

        logging.info(f"Model registered successfully! Model info: {model_info}")
        print(f"Model registered successfully!")
        
        # Save model info to output path
        logging.info(f"Saving model info to: {args.model_info_output_path}")
        os.makedirs(args.model_info_output_path, exist_ok=True)
        model_info_dict = {
            "model_uri": model_info.model_uri,
            "run_id": model_info.run_id,
            "model_name": args.model_name
        }
        with open(os.path.join(args.model_info_output_path, "model_info.json"), "w") as f:
            json.dump(model_info_dict, f, indent=2)
        
        logging.info("Model info saved successfully!")

    except Exception as e:
        logging.error(f"Error during model registration: {str(e)}", exc_info=True)
        mlflow.end_run()
        raise

    finally:
        # End MLflow run
        mlflow.end_run()


if __name__ == "__main__":
    main()
