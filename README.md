# Meeting Summarization Project

A modular project for fine-tuning and evaluating models for meeting summarization.

## Project Structure

- `data/`: Datasets (raw, processed, interim)
- `src/`: Core logic for data processing, modeling, and evaluation
- `configs/`: YAML configuration files
- `notebooks/`: Exploration and validation notebooks
- `outputs/`: Training logs, checkpoints, and predictions

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare data:
   Add raw datasets to `data/raw/` and run the data pipeline.
3. Train model:
   ```bash
   python main.py --mode train
   ```
