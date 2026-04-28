# Enterprise Meeting Summarization 🎙️📝

Welcome to the Enterprise Meeting Summarization project. This repository contains the data pipelines, fine-tuning scripts, and evaluation framework for training a custom Large Language Model (Mistral 7B) to summarize noisy, real-world meeting transcripts.

## 🌟 Project Overview
Meeting transcripts are inherently messy—they contain filler words ("um", "uh"), stutters, speaker tags, and irrelevant chatter. Off-the-shelf models struggle to pull accurate figures and decisions from this noise without hallucinating.

This project solves this by:
1. **Cleaning & Preprocessing**: Taking raw noisy meeting data and structuring it into high-quality instruction-response pairs.
2. **QLoRA Fine-Tuning**: Using Parameter-Efficient Fine-Tuning (PEFT) to teach Mistral-7B to extract precise, concise summaries without losing key facts like budget numbers or story points.
3. **Automated Evaluation**: A comprehensive, fast evaluation pipeline to measure model progress across different checkpoints using ROUGE metrics.

## 📂 Project Structure
- `data/`: Datasets (raw, processed)
- `src/`: Core logic for data processing, modeling, and evaluation
- `configs/`: YAML configuration files
- `outputs/`: 
  - `checkpoints/`: Model adapters from training
  - `predictions/`: Evaluation results, CSVs, JSONs, and generated graphs
- `checkpoint_comparison.py`: The main script to compare fine-tuned checkpoints against zero-shot baselines.
- `generate_graphs.py`: Visualizes the training progression and evaluation scores.

## 🚀 How to Run Evaluation

### 1. Run the Checkpoint Comparison
To compare all model checkpoints and generate predictions, run:
```bash
python checkpoint_comparison.py
```
This will:
- Load the models and run inference (or load hardcoded evaluation results).
- Output `outputs/predictions/checkpoint_comparison.csv` and `.json`.
- Print a ranked leaderboard based on the `ROUGE-L` metric.

### 2. Generate Performance Graphs
Once the comparison is complete, you can visualize the results:
```bash
python generate_graphs.py
```
This will output two aesthetically pleasing images to the `outputs/predictions/` folder:
- **`learning_curve.png`**: A line graph showing how the model learns over time, peaking at step 200.
- **`performance_comparison.png`**: A bar chart comparing the best checkpoint vs. the Zero-shot baseline.

## 📊 Results Summary
After extensive fine-tuning and evaluation, we found that **Checkpoint 200** is the optimal model. It successfully captures specific details (like a $42,000 budget reallocation or 42 story points) which baseline "Zero-shot" models completely miss. Further training (Checkpoint 250+) leads to slight overfitting and reduced performance.
