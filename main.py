import argparse
import json
import sys
import torch
import numpy as np
import os
from gmda.models import GMDARunner
from gmda.data_utils import GestureLoader, DiabetesLoader, MagicLoader, WILTLoader
from gmda.models.gmda.tools import get_config
from gmda.metrics import get_corr_error, get_precision_recall

def parse_arguments():
    parser = argparse.ArgumentParser(description='GMDA Training and Evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--path_train', type=str, required=True, help='Path to train csv file.')
    parser.add_argument('--path_test', type=str, required=True, help='Path to test csv file.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--config', type=str, required=True, help='Path to the config JSON file')
    parser.add_argument('--compute_metrics', action='store_true', help='Whether to compute metrics')
    parser.add_argument('--save_generated', action='store_true', help='Whether to save generated data')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    
    # Add any other arguments you need
    return parser, parser.parse_args()

def train_generator(args):
    # Load configuration
    config = get_config(args.config)    

    # Device
    config['device'] = args.device
    config['model']['device'] = args.device

    # Load and preprocess data
    dataprocessors = {"diabetes": DiabetesLoader(),
                    "gesture": GestureLoader(),
                    "magic": MagicLoader(),
                    "wilt": WILTLoader(),
                    }
    
    processor = dataprocessors[args.dataset]
    processor.init_path(args.path_train, args.path_test)
    train_loader, val_loader, X, y = processor.train_val_loaders(
                                                                batch_size=config['training']['batch_size'], 
                                                                density=config['training']['density'],
                                                                )       

    # Initialize and train the GMDARunner
    runner = GMDARunner(config)
    runner.train(train_loader, val_loader, X, config['training'])

    # Generate synthetic data
    X_synthetic, y_synthetic = runner.generate(y)
    X_synthetic, y_synthetic = X_synthetic.numpy(), y_synthetic.numpy()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save generated data if requested
    if args.save_generated:
        np.save(os.path.join(args.output_dir, 'synthetic_data.npy'), X_synthetic)
        np.save(os.path.join(args.output_dir, 'labels.npy'), y_synthetic)
        print(f"-----\nSynthetic data saved to {os.path.join(args.output_dir, 'synthetic_data.npy')}.\n-----")

    # Compute metrics if requested
    if args.compute_metrics:
        X = X.numpy()
        # Correlation error
        idx = np.random.choice(np.arange(len(X)), size=(min(len(X), 1500),), replace=False) # Subset of real and fake data to avoid memory overload
        corr_error, corr_error_matrix = get_corr_error(X[idx], X_synthetic[idx])
        print(f"-----\nCorr. error:{corr_error}\n-----")

        # Precision/Recall
        precision, recall = get_precision_recall(X, X_synthetic, nb_nn=config['training']['nb_nn_for_prec_recall'])
        print(f"-----\nPrecision: {precision} || Recall: {recall}\n-----")

        # Optionally save metrics
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        metrics = {'correlation error': corr_error,
                   'precision': precision,
                   'recall': recall}
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        print(f"-----\nMetrics saved to {metrics_path}\n-----")
    
def main():
    # Parse command-line arguments
    parser, args = parse_arguments()

    # Train and evaluate
    train_generator(args)

if __name__ == "__main__":
    main()