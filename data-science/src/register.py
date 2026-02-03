import argparse
from pathlib import Path
import mlflow
import os 
import json


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

    # Load model
    model = mlflow.sklearn.load_model(args.model_path)

    # Log and register model with MLflow
    mlflow.sklearn.log_model(model, args.model_name)

    run_id = mlflow.active_run().info.run_id
    model_uri = f'runs:/{run_id}/{args.model_name}'
    mlflow_model = mlflow.register_model(model_uri, args.model_name)
    model_version = mlflow_model.version

    # Handle output path
    try:
        output_path = Path(args.model_info_output_path) / "model_info.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Insufficient permissions for {args.model_info_output_path}. Using /tmp instead.")
        output_path = Path("/tmp/model_info.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write model info
    print("Writing JSON")
    model_info = {"id": f"{args.model_name}:{model_version}"}
    with open(output_path, "w") as of:
        json.dump(model_info, of)

    print(f"Model information saved successfully at {output_path}")


if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
