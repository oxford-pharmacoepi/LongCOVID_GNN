#!/usr/bin/env python3
"""
Unified Experiment + Validation Pipeline
Systematically tests different configurations with proper validation metrics.
Tests: negative sampling strategies, loss functions, and pos:neg ratios.
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import datetime as dt
import sys
import os

from src.config import get_config

def update_config_file(strategy, train_neg_ratio, pos_neg_ratio, loss_function, model_choice='Transformer', primary_metric='apr'):
    """Update the config.py file with new parameters."""
    config_path = Path(__file__).parent / "src" / "config.py"
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Get current config to replace
    current_config = get_config()
    
    # Update negative sampling strategy
    old_strategy = f"self.negative_sampling_strategy = '{current_config.negative_sampling_strategy}'"
    new_strategy = f"self.negative_sampling_strategy = '{strategy}'"
    content = content.replace(old_strategy, new_strategy)
    
    # Update train_neg_ratio
    old_train_ratio = f"self.train_neg_ratio = {current_config.train_neg_ratio}"
    new_train_ratio = f"self.train_neg_ratio = {train_neg_ratio}"
    content = content.replace(old_train_ratio, new_train_ratio)
    
    # Update pos_neg_ratio (val/test)
    old_ratio = f"self.pos_neg_ratio = {current_config.pos_neg_ratio}"
    new_ratio = f"self.pos_neg_ratio = {pos_neg_ratio}"
    content = content.replace(old_ratio, new_ratio)
    
    # Update loss function
    old_loss = f"self.loss_function = '{current_config.loss_function}'"
    new_loss = f"self.loss_function = '{loss_function}'"
    content = content.replace(old_loss, new_loss)
    
    # Update model choice
    old_model = f"self.model_choice = '{current_config.model_choice}'"
    new_model = f"self.model_choice = '{model_choice}'"
    content = content.replace(old_model, new_model)
    
    # Update primary metric
    old_metric = f"self.primary_metric = '{current_config.primary_metric}'"
    new_metric = f"self.primary_metric = '{primary_metric}'"
    content = content.replace(old_metric, new_metric)
    
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"Config updated: strategy={strategy}, train_ratio=1:{train_neg_ratio}, val/test_ratio=1:{pos_neg_ratio}, loss={loss_function}, model={model_choice}, metric={primary_metric}")


def run_experiment(exp_name, strategy, train_neg_ratio, pos_neg_ratio, loss_function, model_choice='Transformer', primary_metric='apr'):
    """Run a single experiment using the three pipeline scripts."""
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_name}")
    print(f"Strategy: {strategy}, Train Ratio: 1:{train_neg_ratio}, Val/Test Ratio: 1:{pos_neg_ratio}")
    print(f"Loss: {loss_function}, Model: {model_choice}, Primary Metric: {primary_metric}")
    print("="*80 + "\n")
    
    # Update config
    update_config_file(strategy, train_neg_ratio, pos_neg_ratio, loss_function, model_choice, primary_metric)
    
    # Force reload config module
    import importlib
    import src.config
    importlib.reload(src.config)
    
    try:
        # Step 1: Create graph
        print("[1/3] Creating graph...")
        result = subprocess.run(
            ["python3", "scripts/1_create_graph.py", 
             "--experiment-name", f"exp_{exp_name}"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"ERROR: Graph creation failed:")
            print(result.stderr[-500:])
            return None
        
        print("Graph created successfully")
        
        # Step 2: Train models
        print("[2/3] Training models...")
        result = subprocess.run(
            ["python3", "scripts/2_train_models.py"],
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        if result.returncode != 0:
            print(f"ERROR: Training failed:")
            print(result.stderr[-500:])
            return None
        
        print("Training completed successfully")
        
        # Step 3: Test and evaluate models
        print("[3/3] Testing and evaluating models...")
        result = subprocess.run(
            ["python3", "scripts/3_test_evaluate.py",
             "--no-mlflow"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"ERROR: Evaluation failed:")
            print(result.stderr[-500:])
            return None
        
        print("Evaluation completed successfully")
        
        # Parse results from test_results_summary_*.csv
        results_dir = Path('results/evaluation')
        if not results_dir.exists():
            print("ERROR: No evaluation results found")
            return None
        
        result_files = list(results_dir.glob('test_results_summary_*.csv'))
        if not result_files:
            print("ERROR: No summary results file found")
            return None
        
        latest_results = max(result_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest_results)
        
        # Convert to dictionary format
        results = {}
        for _, row in df.iterrows():
            model_name = row['Model']
            results[model_name] = {
                'auc': row['auc'],
                'apr': row['apr'],
                'f1': row['f1'],
                'accuracy': row['accuracy'],
                'precision': row['precision'],
                'recall': row['recall'],
                'specificity': row['specificity']
            }
            print(f"  {model_name}: APR={row['apr']:.4f}, AUC={row['auc']:.4f}, F1={row['f1']:.4f}")
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Experiment timed out")
        return None
    except Exception as e:
        print(f"ERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_experiment_grid():
    """Define the experiment grid to test."""
    
    experiments = []
    
    print("="*80)
    print("EXPERIMENT CATEGORIES")
    print("="*80)
    
    # 1. Test different negative sampling strategies (baseline: 1:10 ratio)
    print("\n[1] NEGATIVE SAMPLING STRATEGIES (ratio 1:10)")
    for strategy in ['random', 'hard', 'degree_matched']:
        experiments.append({
            'name': f'strategy_{strategy}',
            'strategy': strategy,
            'train_ratio': 10,
            'val_test_ratio': 10,
            'loss': 'standard_bce',
            'model': 'Transformer'
        })
    
    # 2. Test different ratios with hard sampling
    print("\n[2] NEGATIVE RATIOS WITH HARD SAMPLING")
    for ratio in [1, 20, 50, 100]:
        experiments.append({
            'name': f'hard_ratio_{ratio}',
            'strategy': 'hard',
            'train_ratio': ratio,
            'val_test_ratio': ratio,
            'loss': 'standard_bce',
            'model': 'Transformer'
        })
    
    # 3. Test one case where train and val/test ratios differ
    print("\n[3] DIFFERENT TRAIN vs VAL/TEST RATIO (1:10 vs 1:20)")
    experiments.append({
        'name': 'different_ratios_10_20',
        'strategy': 'hard',
        'train_ratio': 10,
        'val_test_ratio': 20,
        'loss': 'standard_bce',
        'model': 'Transformer'
    })
    
    # 4. Test different loss functions (with optimal: hard, 1:10)
    print("\n[4] LOSS FUNCTIONS (hard, 1:10)")
    for loss in ['weighted_bce', 'focal', 'balanced_focal']:
        experiments.append({
            'name': f'loss_{loss}',
            'strategy': 'hard',
            'train_ratio': 10,
            'val_test_ratio': 10,
            'loss': loss,
            'model': 'Transformer'
        })
    
    # 5. Test all models with best config (hard, 1:10, standard_bce)
    print("\n[5] MODEL ARCHITECTURES (hard, 1:10)")
    for model in ['GCN', 'SAGE']:
        experiments.append({
            'name': f'model_{model}',
            'strategy': 'hard',
            'train_ratio': 10,
            'val_test_ratio': 10,
            'loss': 'standard_bce',
            'model': model
        })
    
    # 6. Test different primary metrics (AUC vs F1 vs APR)
    print("\n[6] PRIMARY METRICS (hard, 1:10)")
    for metric in ['auc', 'f1']:
        experiments.append({
            'name': f'metric_{metric}',
            'strategy': 'hard',
            'train_ratio': 10,
            'val_test_ratio': 10,
            'loss': 'standard_bce',
            'model': 'Transformer',
            'primary_metric': metric
        })
    
    print(f"\n{'='*80}")
    print(f"TOTAL EXPERIMENTS: {len(experiments)}")
    print(f"{'='*80}\n")
    
    return experiments


def main():
    """Run systematic experiments with validation."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENT PIPELINE")
    print("Testing: Negative Sampling | Train/Val/Test Ratios | Loss Functions | Models")
    print("Pipeline: 1_create_graph.py -> 2_train_models.py -> 3_test_evaluate.py")
    print("="*80 + "\n")
    
    # Define experiment grid
    experiments = create_experiment_grid()
    
    print(f"\nTotal experiments to run: {len(experiments)}\n")
    print("Experiment Grid:")
    print("-" * 100)
    for i, exp in enumerate(experiments, 1):
        print(f"{i:2d}. {exp['name']:25s} | strategy={exp['strategy']:15s} | train=1:{exp['train_ratio']:2d} | val/test=1:{exp['val_test_ratio']:2d} | loss={exp['loss']:15s} | model={exp['model']}")
    print("-" * 100)
    
    input("\nPress Enter to start experiments (or Ctrl+C to cancel)...")
    
    # Track all results
    all_results = []
    
    # Run each experiment
    start_time = dt.datetime.now()
    
    # Create results/experiments directory
    experiments_dir = Path('results/experiments')
    experiments_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiments directory: {experiments_dir}")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"Running Experiment {i}/{len(experiments)}: {exp['name']}")
        print(f"{'='*80}")
        
        exp_start = dt.datetime.now()
        
        results = run_experiment(
            exp['name'],
            exp['strategy'],
            exp['train_ratio'],
            exp['val_test_ratio'],
            exp['loss'],
            exp['model'],
            exp.get('primary_metric', 'apr')
        )
        
        exp_duration = (dt.datetime.now() - exp_start).total_seconds()
        
        if results:
            # Flatten results for CSV
            for model_name, metrics in results.items():
                record = {
                    'experiment': exp['name'],
                    'strategy': exp['strategy'],
                    'train_ratio': exp['train_ratio'],
                    'val_test_ratio': exp['val_test_ratio'],
                    'loss': exp['loss'],
                    'model': model_name,
                    'auc': metrics['auc'],
                    'apr': metrics['apr'],
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'specificity': metrics['specificity'],
                    'duration_sec': exp_duration,
                    'timestamp': dt.datetime.now().isoformat()
                }
                all_results.append(record)
            
            # Print summary
            print("\n" + "-"*80)
            print("EXPERIMENT RESULTS:")
            for model_name, metrics in results.items():
                print(f"  {model_name:15s}: APR={metrics['apr']:.4f}, AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
            print("-"*80)
        
        # Save intermediate results
        if all_results:
            df = pd.DataFrame(all_results)
            temp_file = "experiment_results_temp.csv"
            df.to_csv(temp_file, index=False)
            print(f"Intermediate results saved to {temp_file}")
    
    total_duration = (dt.datetime.now() - start_time).total_seconds()
    
    # Save final results
    if all_results:
        df = pd.DataFrame(all_results)
        
        output_file = experiments_dir / f"experiment_results_comprehensive_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*80)
        print(f"\nTotal duration: {total_duration/60:.1f} minutes")
        print(f"Results saved to: {output_file}")
        
        # Create summary tables
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY")
        print("="*80)
        
        # 1. Top 20 by APR
        print("\nTOP 20 CONFIGURATIONS BY APR:")
        print("-"*100)
        top_configs = df.nlargest(20, 'apr')[['experiment', 'strategy', 'train_ratio', 'val_test_ratio', 'loss', 'model', 'apr', 'auc', 'f1']]
        print(top_configs.to_string(index=False))
        
        # 2. Strategy comparison
        print("\n\nPERFORMANCE BY STRATEGY:")
        print("-"*80)
        strategy_avg = df.groupby('strategy')[['auc', 'apr', 'f1', 'precision', 'recall']].agg(['mean', 'std'])
        print(strategy_avg.to_string())
        
        # 3. Loss function comparison
        print("\n\nPERFORMANCE BY LOSS FUNCTION:")
        print("-"*80)
        loss_avg = df.groupby('loss')[['auc', 'apr', 'f1', 'precision', 'recall']].agg(['mean', 'std'])
        print(loss_avg.to_string())
        
        # 4. Model comparison
        print("\n\nPERFORMANCE BY MODEL:")
        print("-"*80)
        model_avg = df.groupby('model')[['auc', 'apr', 'f1', 'precision', 'recall']].agg(['mean', 'std'])
        print(model_avg.to_string())
        
        # 5. Train ratio impact
        print("\n\nIMPACT OF TRAINING RATIO:")
        print("-"*80)
        train_ratio_avg = df.groupby('train_ratio')[['auc', 'apr', 'f1']].mean()
        print(train_ratio_avg.to_string())
        
        # 6. Val/Test ratio impact
        print("\n\nIMPACT OF VAL/TEST RATIO:")
        print("-"*80)
        val_test_ratio_avg = df.groupby('val_test_ratio')[['auc', 'apr', 'f1']].mean()
        print(val_test_ratio_avg.to_string())
        
        # 7. Best overall
        best_idx = df['apr'].idxmax()
        best_config = df.loc[best_idx]
        
        print("\n\nBEST OVERALL CONFIGURATION:")
        print("="*80)
        print(f"  Experiment:        {best_config['experiment']}")
        print(f"  Strategy:          {best_config['strategy']}")
        print(f"  Train Ratio:       1:{best_config['train_ratio']}")
        print(f"  Val/Test Ratio:    1:{best_config['val_test_ratio']}")
        print(f"  Loss:              {best_config['loss']}")
        print(f"  Model:             {best_config['model']}")
        print(f"\n  METRICS:")
        print(f"    APR:             {best_config['apr']:.4f}")
        print(f"    AUC:             {best_config['auc']:.4f}")
        print(f"    F1:              {best_config['f1']:.4f}")
        print(f"    Precision:       {best_config['precision']:.4f}")
        print(f"    Recall:          {best_config['recall']:.4f}")
        print("="*80)
        
        # Create visualisations
        print("\nCreating comparison visualisations...")
        create_comparison_plots(df, output_file.replace('.csv', ''))
        
        print(f"\nAll results saved to: {output_file}")
        print(f"Visualisations saved with prefix: {output_file.replace('.csv', '')}_*")
        
    else:
        print("\nERROR: No experiments completed successfully")


def create_comparison_plots(df, output_prefix):
    """Create comprehensive comparison visualisations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    
    # 1. Top 20 by APR
    fig, ax = plt.subplots(figsize=(14, 8))
    df_sorted = df.sort_values('apr', ascending=False).head(20)
    colors = ['#2ecc71' if apr > 0.6 else '#f39c12' if apr > 0.4 else '#e74c3c' for apr in df_sorted['apr']]
    ax.barh(df_sorted['experiment'], df_sorted['apr'], color=colors, alpha=0.8)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='APR = 0.5')
    ax.axvline(x=0.6, color='green', linestyle='--', alpha=0.5, label='APR = 0.6')
    ax.set_xlabel('Average Precision (APR)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Configurations by APR', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top20_apr.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Strategy comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(['apr', 'auc', 'f1']):
        df.boxplot(column=metric, by='strategy', ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by Strategy', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Strategy')
        axes[i].set_ylabel(metric.upper())
    plt.suptitle('Performance by Negative Sampling Strategy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Loss comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(['apr', 'auc', 'f1']):
        df.boxplot(column=metric, by='loss', ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by Loss', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Loss Function')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
    plt.suptitle('Performance by Loss Function', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_metrics = df.groupby('model')[['apr', 'auc', 'f1']].mean()
    model_metrics.plot(kind='bar', ax=ax, rot=0, alpha=0.8)
    ax.set_title('Average Performance by Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(title='Metric', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Ratio impact
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train ratio
    train_impact = df.groupby('train_ratio')[['apr', 'auc', 'f1']].mean()
    train_impact.plot(ax=axes[0], marker='o', linewidth=2)
    axes[0].set_title('Impact of Training Ratio', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Training Neg:Pos Ratio (1:X)', fontsize=10)
    axes[0].set_ylabel('Score', fontsize=10)
    axes[0].legend(title='Metric')
    axes[0].grid(alpha=0.3)
    
    # Val/Test ratio
    valtest_impact = df.groupby('val_test_ratio')[['apr', 'auc', 'f1']].mean()
    valtest_impact.plot(ax=axes[1], marker='s', linewidth=2)
    axes[1].set_title('Impact of Val/Test Ratio', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Val/Test Neg:Pos Ratio (1:X)', fontsize=10)
    axes[1].set_ylabel('Score', fontsize=10)
    axes[1].legend(title='Metric')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_ratio_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = df[['train_ratio', 'val_test_ratio', 'auc', 'apr', 'f1', 'precision', 'recall', 'specificity']].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualisations created")


if __name__ == "__main__":
    main()
