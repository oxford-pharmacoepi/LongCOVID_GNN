"""
Evaluation metrics and statistical utilities.
"""

import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score, confusion_matrix

def calculate_bootstrap_ci(y_true, y_pred_proba, y_pred_binary, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for classification metrics."""
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Storage for bootstrap results
    bootstrap_metrics = {
        'sensitivity': [], 'specificity': [], 'precision': [], 
        'f1': [], 'auc': [], 'apr': []
    }
    
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    
    for i in range(n_bootstrap):
        # Bootstrap sample indices
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Bootstrap samples
        boot_y_true = y_true[boot_indices]
        boot_y_pred_proba = y_pred_proba[boot_indices] 
        boot_y_pred_binary = y_pred_binary[boot_indices]
        
        # Calculate metrics for this bootstrap sample
        # Confusion matrix elements
        tp = np.sum(boot_y_pred_binary * boot_y_true)
        fp = np.sum(boot_y_pred_binary * (1 - boot_y_true))
        fn = np.sum((1 - boot_y_pred_binary) * boot_y_true)
        tn = np.sum((1 - boot_y_pred_binary) * (1 - boot_y_true))
        
        # Classification metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Store results
        bootstrap_metrics['sensitivity'].append(sensitivity)
        bootstrap_metrics['specificity'].append(specificity) 
        bootstrap_metrics['precision'].append(precision)
        bootstrap_metrics['f1'].append(f1)
        
        # AUC and APR (only if we have both classes)
        if len(np.unique(boot_y_true)) == 2:
            try:
                auc_score = roc_auc_score(boot_y_true, boot_y_pred_proba)
                apr_score = average_precision_score(boot_y_true, boot_y_pred_proba)
                bootstrap_metrics['auc'].append(auc_score)
                bootstrap_metrics['apr'].append(apr_score)
            except:
                # Skip this iteration if AUC/APR calculation fails
                bootstrap_metrics['auc'].append(np.nan)
                bootstrap_metrics['apr'].append(np.nan)
        else:
            bootstrap_metrics['auc'].append(np.nan)
            bootstrap_metrics['apr'].append(np.nan)
    
    # Calculate confidence intervals
    ci_results = {}
    for metric, values in bootstrap_metrics.items():
        # Remove NaN values
        clean_values = [v for v in values if not np.isnan(v)]
        
        if len(clean_values) > 0:
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            ci_lower = np.percentile(clean_values, lower_percentile)
            ci_upper = np.percentile(clean_values, upper_percentile)
            mean_val = np.mean(clean_values)
            
            ci_results[metric] = {
                'mean': mean_val,
                'ci_lower': ci_lower, 
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower
            }
        else:
            ci_results[metric] = {
                'mean': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan, 
                'ci_width': np.nan
            }
    
    return ci_results


def calculate_metrics(y_true, y_prob, y_pred, k_values=[10, 50, 100, 200, 500]):
    """Calculate comprehensive evaluation metrics including Recall@K."""
    
    # Basic metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Advanced metrics
    auc_score = roc_auc_score(y_true, y_prob)
    apr_score = average_precision_score(y_true, y_prob)
    
    # Ranking metrics
    ranking_metrics = calculate_ranking_metrics(y_true, y_prob, k_values)
    
    # Additional metrics
    ppv = precision  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Calculate bootstrap confidence intervals
    ci_results = calculate_bootstrap_ci(y_true, y_prob, y_pred, n_bootstrap=1000)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc_score,
        'apr': apr_score,
        'ppv': ppv,
        'npv': npv,
        'confusion_matrix': {
            'TP': int(tp), 'FP': int(fp), 
            'TN': int(tn), 'FN': int(fn)
        },
        'ci_results': ci_results,
        'ranking_metrics': ranking_metrics,
        **ranking_metrics # Flatten ranking metrics into main dict for easier access
    }


def calculate_recall_at_k(y_true, y_prob, k_values=[10, 50, 100, 200, 500]):
    """Calculate Recall@K for ranking-based evaluation."""
    # Get indices of true positives
    true_positives = np.where(y_true == 1)[0]
    n_true_positives = len(true_positives)
    
    if n_true_positives == 0:
        return {f'recall@{k}': 0.0 for k in k_values}
    
    # Sort predictions by probability (descending)
    sorted_indices = np.argsort(y_prob)[::-1]
    
    recall_at_k = {}
    for k in k_values:
        # Get top-K predictions
        top_k_indices = sorted_indices[:k]
        
        # Count how many true positives are in top-K
        true_positives_in_top_k = np.sum(y_true[top_k_indices] == 1)
        
        # Calculate recall
        recall = true_positives_in_top_k / n_true_positives
        recall_at_k[f'recall@{k}'] = recall
    
    return recall_at_k


def calculate_hits_at_k(y_true, y_prob, k_values=[10, 50, 100, 200, 500]):
    """
    Calculate Hits@K for ranking-based evaluation.
    Hits@K = 1 if at least one positive is in top K, else 0.
    """
    # Sort predictions by probability (descending)
    sorted_indices = np.argsort(y_prob)[::-1]
    
    hits_at_k = {}
    for k in k_values:
        # Get top-K predictions
        top_k_indices = sorted_indices[:k]
        
        # Check if at least one true positive is in top-K
        hit = 1 if np.sum(y_true[top_k_indices] == 1) > 0 else 0
        hits_at_k[f'hits_at_{k}'] = float(hit)
    
    return hits_at_k


def calculate_precision_at_k(y_true, y_prob, k_values=[10, 50, 100, 200, 500]):
    """Calculate Precision@K for ranking-based evaluation."""
    # Sort predictions by probability (descending)
    sorted_indices = np.argsort(y_prob)[::-1]
    
    precision_at_k = {}
    for k in k_values:
        # Get top-K predictions
        top_k_indices = sorted_indices[:k]
        
        # Count how many are true positives
        true_positives_in_top_k = np.sum(y_true[top_k_indices] == 1)
        
        # Calculate precision
        precision = true_positives_in_top_k / k if k > 0 else 0
        precision_at_k[f'precision@{k}'] = precision
    
    return precision_at_k


def calculate_ndcg_at_k(y_true, y_prob, k_values=[10, 50, 100, 200, 500]):
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
    ndcg_at_k = {}
    for k in k_values:
        try:
            # Calculate NDCG@K
            score = ndcg_score(y_true.reshape(1, -1), y_prob.reshape(1, -1), k=k)
            ndcg_at_k[f'ndcg@{k}'] = score
        except:
            ndcg_at_k[f'ndcg@{k}'] = 0.0
    
    return ndcg_at_k


def calculate_ranking_metrics(y_true, y_prob, k_values=[10, 50, 100, 200, 500]):
    """Calculate comprehensive ranking metrics for drug repurposing evaluation."""
    metrics = {}
    
    # Calculate Recall@K
    metrics.update(calculate_recall_at_k(y_true, y_prob, k_values))
    
    # Calculate Precision@K
    metrics.update(calculate_precision_at_k(y_true, y_prob, k_values))
    
    # Calculate Hits@K
    metrics.update(calculate_hits_at_k(y_true, y_prob, k_values))
    
    # Calculate NDCG@K
    metrics.update(calculate_ndcg_at_k(y_true, y_prob, k_values))
    
    return metrics
