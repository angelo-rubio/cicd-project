import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def main():
    """Data preparation job: encodes categorical features and splits data into train/test sets."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare data for model training")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input CSV data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Ratio for test split")
    parser.add_argument("--train_data", type=str, required=True, help="Path to save training data")
    parser.add_argument("--test_data", type=str, required=True, help="Path to save testing data")
    args = parser.parse_args()

    # Start MLflow logging
    mlflow.start_run()

    try:
        # Log arguments
        logging.info(f"Input data path: {args.input_data}")
        logging.info(f"Test-train ratio: {args.test_train_ratio}")

        # Read the input data
        logging.info("Reading input data...")
        df = pd.read_csv(args.input_data)
        logging.info(f"Data shape: {df.shape}")

        # Encode categorical 'Segment' column
        logging.info("Encoding 'Segment' column...")
        label_encoder = LabelEncoder()
        df['Segment'] = label_encoder.fit_transform(df['Segment'])
        logging.info(f"Encoded segments: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

        # Log first few rows of transformed data
        logging.info(f"Transformed Data:\n{df.head()}")

        # Split the data
        logging.info("Splitting data into train/test sets...")
        train_df, test_df = train_test_split(
            df, 
            test_size=args.test_train_ratio, 
            random_state=21
        )

        # Log the number of records
        train_records = len(train_df)
        test_records = len(test_df)
        print(f"Training records: {train_records}")
        print(f"Testing records: {test_records}")
        
        logging.info(f"Training records: {train_records}")
        logging.info(f"Testing records: {test_records}")

        # Log metrics to MLflow
        mlflow.log_metric("train_records", train_records)
        mlflow.log_metric("test_records", test_records)

        # Create output directories if they don't exist
        os.makedirs(args.train_data, exist_ok=True)
        os.makedirs(args.test_data, exist_ok=True)

        # Save the split data
        train_path = os.path.join(args.train_data, "train.csv")
        test_path = os.path.join(args.test_data, "test.csv")

        logging.info(f"Saving training data to: {train_path}")
        train_df.to_csv(train_path, index=False)
        
        logging.info(f"Saving testing data to: {test_path}")
        test_df.to_csv(test_path, index=False)

        print(f"Training data saved to: {train_path}")
        print(f"Testing data saved to: {test_path}")
        
        logging.info("Data preparation completed successfully!")

    except Exception as e:
        logging.error(f"Error during data preparation: {str(e)}", exc_info=True)
        mlflow.end_run()
        raise

    finally:
        # End MLflow run
        mlflow.end_run()


if __name__ == "__main__":
    main()
