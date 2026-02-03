import os
import argparse
import logging
import mlflow
from pathlib import Path
import json


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Register trained model in MLflow")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the registered model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--model_info_output_path", type=str, required=True, help="Path to save model info")
    args = parser.parse_args()

    mlflow.start_run()

    try:
        logging.info(f"Loading model from: {args.model_path}")
        model = mlflow.sklearn.load_model(args.model_path)

        logging.info(f"Registering model '{args.model_name}' in MLflow...")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_price_regressor",
            registered_model_name=args.model_name
        )

        logging.info("Model registered successfully!")
        
        output_dir = Path(args.model_info_output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_info_dict = {
            "model_uri": model_info.model_uri,
            "run_id": model_info.run_id,
            "model_name": args.model_name
        }
        
        output_file = output_dir / "model_info.json"
        with open(output_file, "w") as f:
            json.dump(model_info_dict, f, indent=2)

        logging.info(f"Model info saved to {output_file}")

    except Exception as e:
        logging.error(f"Error during model registration: {str(e)}", exc_info=True)
        raise

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
