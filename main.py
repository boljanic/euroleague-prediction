"""
Pipeline orchestrator.

Usage:
    python main.py prepare [--data path/to/file.csv]
    python main.py train
    python main.py evaluate
    python main.py all [--data path/to/file.csv]
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def step_prepare(data_path):
    from src.data_preparation import run_pipeline
    run_pipeline(data_path)


def step_train():
    from src.train import main as train_main
    train_main()


def step_evaluate():
    from src.evaluate import main as evaluate_main
    evaluate_main()


def main():
    parser = argparse.ArgumentParser(description='Euroleague ML pipeline')
    parser.add_argument('step', choices=['prepare', 'train', 'evaluate', 'all'])
    parser.add_argument('--data', default=None,
                        help='Path to the raw Header CSV (required for prepare/all)')
    args = parser.parse_args()

    if args.step in ('prepare', 'all'):
        print("\n" + "="*60)
        print("  STEP 1 — DATA PREPARATION")
        print("="*60)
        step_prepare(args.data)

    if args.step in ('train', 'all'):
        print("\n" + "="*60)
        print("  STEP 2 — MODEL TRAINING")
        print("="*60)
        step_train()

    if args.step in ('evaluate', 'all'):
        print("\n" + "="*60)
        print("  STEP 3 — EVALUATION & VISUALISATION")
        print("="*60)
        step_evaluate()

    print("\nPipeline finished.")
    if args.step in ('evaluate', 'all'):
        print("   Figures  ->  results/figures/")
        print("   Metrics  ->  results/metrics/")
        print("\nTo launch the web app:")
        print("   streamlit run app/app.py")


if __name__ == '__main__':
    main()
