import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_dynamics(csv_file):
    df = pd.read_csv(csv_file)
    
    # Create figure with 2 y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Training Loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    sns.lineplot(data=df, x='epoch', y='train_loss', ax=ax1, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second axis for Rank
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Median Rank (Lower is Better)', color=color)
    # Filter out missing rank data (-1)
    rank_data = df[df['val_rank_median'] > 0]
    sns.lineplot(data=rank_data, x='epoch', y='val_rank_median', ax=ax2, color=color, marker='o', label='Median Rank')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Dynamics: Loss vs Validation Rank')
    fig.tight_layout()
    plt.savefig('results/learning_curve.png')
    print("Plot saved to results/learning_curve.png")
    
    # Print numerical summary
    best_epoch = rank_data.loc[rank_data['val_rank_median'].idxmin()]
    print("\nBest Validation Performance:")
    print(best_epoch)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_dynamics(sys.argv[1])
    else:
        print("Usage: python plot_learning_curve.py <csv_file>")
