# Main entry point for the meeting summarization project
import argparse
from src.pipeline.train_pipeline import run_train_pipeline
from src.pipeline.data_pipeline import run_data_pipeline

def main():
    parser = argparse.ArgumentParser(description="Meeting Summarization Pipeline")
    parser.add_argument("--mode", type=str, choices=["data", "train", "eval"], help="Mode to run")
    args = parser.parse_args()

    if args.mode == "data":
        run_data_pipeline()
    elif args.mode == "train":
        run_train_pipeline()
    elif args.mode == "eval":
        # Add eval pipeline call
        pass
    else:
        print("Please specify a mode: --mode [data|train|eval]")

if __name__ == "__main__":
    main()
