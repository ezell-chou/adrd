import argparse
import json
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Any
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from core.classifier import build_model
from core.utilities_train import CustomDataset, EarlyStopping
from configs.config import TrainConfig

# ========================================================================
# Command-line argument parser
# ========================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate classification models with cross-validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python train_eval.py
  
  # Specify custom paths
  python train_eval.py --train_tensor data/custom --eval_tensor data/custom_eval --save_models data/models --results data/results
        """
    )
    
    parser.add_argument('--train_tensor', type=str, default=None,
                       help='Path to training data directory')
    parser.add_argument('--eval_tensor', type=str, default=None,
                       help='Path to evaluation data directory')
    parser.add_argument('--save_models', type=str, default=None,
                       help='Path to directory for saving models')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to directory for saving results')
    
    return parser.parse_args()


# ========================================================================
# Helper Functions
# ========================================================================
def _init_model(network: str, device: torch.device) -> torch.nn.Module:
    """
    Initialize model with necessary setup.

    Parameters
    ----------
    network : str
        Network architecture name.
    device : torch.device
        Device to place model on.

    Returns
    -------
    model : torch.nn.Module
        Initialized model on device.
    """
    model = build_model(net=network).to(device)
    if hasattr(model, 'init_flat_fc'):
        dummy = torch.zeros(1, 4, 32, 32).to(device)
        model.init_flat_fc(dummy)
    return model


def _load_datasets(
    data_path: str,
    subset_name: str,
    split_ratio: float = 0.7,
) -> Tuple[CustomDataset, CustomDataset]:
    """
    Load training and validation datasets.

    Parameters
    ----------
    data_path : str
        Root data directory path.
    subset_name : str
        Name of data subset folder.
    split_ratio : float, optional
        Train/validation split ratio, by default 0.7.

    Returns
    -------
    train_ds : CustomDataset
        Training dataset.
    val_ds : CustomDataset
        Validation dataset.
    """
    dataset_path = str(Path(data_path) / subset_name)
    train_ds = CustomDataset(dataset_path, split_ratio=split_ratio, is_train=True)
    val_ds = CustomDataset(dataset_path, split_ratio=split_ratio, is_train=False)
    return train_ds, val_ds


def _compute_aggregate_stats(
    metric_values: np.ndarray,
    metric_name: str = 'metric',
) -> Dict[str, float]:
    """
    Compute aggregate statistics (mean and standard deviation).

    Parameters
    ----------
    metric_values : np.ndarray
        Array of metric values.
    metric_name : str, optional
        Name of metric for reference, by default 'metric'.

    Returns
    -------
    stats : Dict[str, float]
        Dictionary containing 'mean' and 'std' keys.
    """
    return {
        'mean': np.mean(metric_values),
        'std': np.std(metric_values, ddof=1) if len(metric_values) > 1 else 0.0
    }


def _save_json_results(
    results: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Save results to JSON file.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary to save.
    output_path : str
        Path to output JSON file.

    Returns
    -------
    None
    """
    def convert(o: Any) -> Any:
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f'\n>>> Results saved -> {output_path}')


# ========================================================================
# Training Single Subset (Multiple Runs)
# ========================================================================
def train_one_subset(
    train_dir: str,
    sub_name: str,
    run_idx: int,
    device: torch.device,
    network: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    split_ratio: float,
    retrain_times: int,
    save_models: str,
) -> None:
    """
    Train model on single data subset.

    Parameters
    ----------
    train_dir : str
        Training data directory.
    sub_name : str
        Name of subset folder.
    run_idx : int
        Run index for tracking.
    device : torch.device
        Device to use for training.
    network : str
        Network architecture name.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Learning rate for optimizer.
    split_ratio : float
        Train/validation split ratio.
    retrain_times : int
        Total number of retraining times (for display).
    save_models : str
        Directory to save models.

    Returns
    -------
    None
    """
    # Load datasets
    train_ds, val_ds = _load_datasets(train_dir, sub_name, split_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = _init_model(network, device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=8)

    print(f'>>> Training {sub_name} (run {run_idx+1}/{retrain_times})  (early stopping patience=8)')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc, _, _, _ = evaluate(model, val_loader, device)
        print(f'[{sub_name}] Epoch {epoch}/{epochs}  loss={running_loss/len(train_loader):.4f}  val_acc={val_acc:.2%}')

        if early_stop(val_acc, model):
            print(f'[{sub_name}] Early stopping at epoch {epoch}, best val_acc={early_stop.best_score:.2%}')
            break
    else:
        print(f'[{sub_name}] Reached max epochs {epochs}, best val_acc={early_stop.best_score:.2%}')

    # Save best weights
    models_dir = Path(save_models)
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / f'{sub_name}_run{run_idx}.pth'

    torch.save(early_stop.best_state, best_path)
    print(f'>>> Best weights saved -> {best_path}\n')


# ========================================================================
# Evaluation
# ========================================================================
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    loader : DataLoader
        Data loader for evaluation.
    device : torch.device
        Device to use for evaluation.

    Returns
    -------
    acc : float
        Accuracy score.
    all_true : np.ndarray
        True labels.
    all_prob : np.ndarray
        Predicted probabilities.
    all_pred : np.ndarray
        Predicted labels (binary).
    """
    model.eval()
    all_prob, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_prob.append(probs.cpu())
        all_true.append(y.cpu())
    all_prob = torch.cat(all_prob).numpy()
    all_true = torch.cat(all_true).numpy()
    all_pred = (all_prob > 0.5).astype(int)
    acc = accuracy_score(all_true, all_pred)
    return acc, all_true, all_prob, all_pred


@torch.no_grad()
def evaluate_with_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_runs: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate model multiple times with statistics.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    loader : DataLoader
        Data loader for evaluation.
    device : torch.device
        Device to use for evaluation.
    n_runs : int, optional
        Number of evaluation runs, by default 5.

    Returns
    -------
    stats : Dict[str, Dict[str, Any]]
        Dictionary containing mean, std, and values for each metric.
    """
    acc_list, ap_list, auc_list = [], [], []
    for i in range(n_runs):
        print(f'    Running evaluation {i+1}/{n_runs}...')
        if hasattr(loader.dataset, 'shuffle_indices'):
            loader.dataset.shuffle_indices()
        acc, y_true, y_prob, _ = evaluate(model, loader, device)
        acc_list.append(acc)
        ap_list.append(average_precision_score(y_true, y_prob))
        auc_list.append(roc_auc_score(y_true, y_prob))
    stats = {}
    for name, values in [('acc', acc_list), ('ap', ap_list), ('auc', auc_list)]:
        stats[name] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
            'values': values
        }
    return stats


# ========================================================================
# Cross-Validation Main Function
# ========================================================================
def cross_validate(
    train_dir: str,
    eval_dir: str,
    train_subs: List[str],
    eval_subs: List[str],
    device: torch.device,
    network: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    split_ratio: float,
    retrain_times: int,
    force_retrain: bool,
    save_models: str,
    results: str,
    n_eval_runs: int = 5,
) -> None:
    """
    Perform cross-validation training and evaluation.

    Parameters
    ----------
    train_dir : str
        Training data directory.
    eval_dir : str
        Evaluation data directory.
    train_subs : List[str]
        List of training subset names.
    eval_subs : List[str]
        List of evaluation subset names.
    device : torch.device
        Device for training and evaluation.
    network : str
        Network architecture name.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Learning rate for optimizer.
    split_ratio : float
        Train/validation split ratio.
    retrain_times : int
        Number of times to retrain each subset.
    force_retrain : bool
        Whether to force retraining.
    save_models : str
        Directory to save models.
    results : str
        Directory to save results.
    n_eval_runs : int, optional
        Number of evaluation runs per model, by default 5.

    Returns
    -------
    None
    """
    train_dir = Path(train_dir)
    eval_dir = Path(eval_dir)
    models_dir = Path(save_models)
    
    print("="*60)
    print(f"Phase 1: Model Training (each subset trained {retrain_times} times)")
    print(f"Training directory: {train_dir}")
    print("="*60)

    # Train all subsets
    for sub in train_subs:
        for run_idx in range(retrain_times):
            best_path = models_dir / f'{sub}_run{run_idx}.pth'
            if best_path.exists() and not force_retrain:
                print(f'{best_path} exists, skipping training')
                continue
            print(f'\nTraining subset: {sub} (run {run_idx+1}/{retrain_times})')
            train_one_subset(
                train_dir, sub, run_idx,
                device=device,
                network=network,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                split_ratio=split_ratio,
                retrain_times=retrain_times,
                save_models=save_models,
            )

    print("\n" + "="*60)
    print(f"Phase 2: Cross-Evaluation")
    print(f"Evaluation directory: {eval_dir}")
    print("="*60)

    all_results = {tr: {va: {} for va in eval_subs} for tr in train_subs}
    # Pre-load all evaluation datasets to avoid repeated file reads
    eval_datasets = {sub: CustomDataset(str(eval_dir / sub), split_ratio=split_ratio, is_train=False) 
                     for sub in eval_subs}
    
    for train_sub in train_subs:
        for run_idx in range(retrain_times):
            print(f'\nLoading model: {train_sub} (run {run_idx+1}/{retrain_times})')

            # Use optimized initialization function
            model = _init_model(network, device)
            model_path = models_dir / f'{train_sub}_run{run_idx}.pth'
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

            for val_sub in eval_subs:
                print(f'  Evaluating: {train_sub} -> {val_sub}')
                val_loader = DataLoader(eval_datasets[val_sub], batch_size=batch_size, shuffle=False)
                stats = evaluate_with_stats(model, val_loader, device, n_runs=n_eval_runs)
                all_results[train_sub][val_sub][run_idx] = stats
                print(f'    acc: {stats["acc"]["mean"]:.2%} ± {stats["acc"]["std"]:.2%}')


    print("\n" + "="*60)
    print("Phase 3: Aggregate statistics across all runs")
    print("="*60)
    final_results = {tr: {va: {} for va in eval_subs} for tr in train_subs}
    for train_sub in train_subs:
        for val_sub in eval_subs:
            # Extract mean values of each metric across runs
            acc_means = [all_results[train_sub][val_sub][run_idx]["acc"]["mean"] for run_idx in range(retrain_times)]
            ap_means  = [all_results[train_sub][val_sub][run_idx]["ap"]["mean"]  for run_idx in range(retrain_times)]
            auc_means = [all_results[train_sub][val_sub][run_idx]["auc"]["mean"] for run_idx in range(retrain_times)]
            
            # Use optimized function to compute aggregate statistics
            final_results[train_sub][val_sub] = {
                "acc": _compute_aggregate_stats(acc_means, 'acc'),
                "ap":  _compute_aggregate_stats(ap_means,  'ap'),
                "auc": _compute_aggregate_stats(auc_means, 'auc')
            }

    # Save training results
    json_path = Path(results) / 'final_results.json'
    _save_json_results(final_results, str(json_path))
    print('\n>>> Training and evaluation completed! To plot results, run: python plot.py')


def main():
    """
    Main pipeline: load configuration, parse arguments, and run cross-validation.
    """
    # Load default configuration
    config = TrainConfig()
    
    # Get parameters from command-line args or config
    args = get_args()
    
    train_tensor = args.train_tensor or config.TRAIN_TENSOR
    eval_tensor = args.eval_tensor or config.EVAL_TENSOR
    save_models = args.save_models or config.SAVE_MODELS
    results = args.results or config.RESULTS
    
    # Convert to Path
    train_tensor = Path(train_tensor)
    eval_tensor = Path(eval_tensor)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("Training and Evaluation Pipeline")
    print("=" * 70)
    print(f"Device:            {config.DEVICE}")
    print(f"Network:           {config.NETWORK}")
    print(f"Epochs:            {config.EPOCHS}")
    print(f"Batch size:        {config.BATCH_SIZE}")
    print(f"Learning rate:     {config.LEARNING_RATE}")
    print(f"Split ratio:       {config.SPLIT_RATIO}")
    print(f"Retrain times:     {config.RETRAIN_TIMES}")
    print(f"Force retrain:     {config.FORCE_RETRAIN}")
    print(f"N eval runs:       {config.N_EVAL_RUNS}")
    print(f"Data directory:    {train_tensor}")
    print(f"Eval directory:    {eval_tensor}")
    print(f"Save model dir:    {save_models}")
    print(f"Results dir:       {results}")
    print("=" * 70 + "\n")

    train_subfolder_names = [n.name for n in train_tensor.iterdir()
                                if n.is_dir()]
    eval_subfolder_names = [n.name for n in eval_tensor.iterdir()
                            if n.is_dir()]

    print(f"Training subsets: {sorted(train_subfolder_names)}")
    print(f"Evaluation subsets: {sorted(eval_subfolder_names)}\n")

    cross_validate(
        train_tensor, eval_tensor, train_subfolder_names, eval_subfolder_names,
        device=config.DEVICE,
        network=config.NETWORK,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        split_ratio=config.SPLIT_RATIO,
        retrain_times=config.RETRAIN_TIMES,
        force_retrain=config.FORCE_RETRAIN,
        save_models=save_models,
        results=results,
        n_eval_runs=config.N_EVAL_RUNS,
    )


if __name__ == "__main__":
    main()