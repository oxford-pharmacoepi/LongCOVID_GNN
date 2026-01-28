#!/usr/bin/env python3
"""
Graph Creation Script for Drug-Disease Prediction
Creates knowledge graph from OpenTargets data with validation/test splits.
"""

import argparse
import datetime as dt
import json
import os
import torch
from pathlib import Path

# Import from shared modules
from src.config import get_config
from src.utils.common import enable_full_reproducibility
from src.utils.graph_utils import standard_graph_analysis
from src.training.tracker import ExperimentTracker
from src.graph.builder import GraphBuilder


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create drug-disease prediction graph')
    parser.add_argument('--output-dir', type=str, default='results/', help='Output directory')
    parser.add_argument('--analyze', action='store_true', help='Run graph analysis')
    parser.add_argument('--force-mode', type=str, choices=['raw', 'processed'], 
                        help='Force specific data processing mode (raw or processed)')
    parser.add_argument('--experiment-name', type=str, default='graph_creation',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Load configuration from config.py
    config = get_config()
    
    # Update output path if specified
    if args.output_dir:
        config.update_paths(results=args.output_dir)
    
    # Set reproducibility
    enable_full_reproducibility(config.seed)
    
    # Initialise MLflow tracker
    tracker = ExperimentTracker(experiment_name=args.experiment_name)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"graph_{config.training_version}_{timestamp}"
    
    try:
        tracker.start_run(run_name=run_name)
        
        # Log configuration
        tracker.log_config(config)
        
        # Log additional graph creation parameters
        tracker.log_param("force_mode", args.force_mode if args.force_mode else "auto")
        tracker.log_param("analyze_graph", args.analyze)
        tracker.log_param("output_dir", args.output_dir)
        
        # Create graph
        builder = GraphBuilder(config, args.force_mode, tracker)
        builder.load_or_create_data()
        builder.create_node_features()
        builder.create_edges()
        builder.create_train_val_test_splits()
        
        graph = builder.build_graph()
        
        # Log graph metadata
        tracker.log_graph_metadata(graph)
        
        # Save graph
        graph_filename = f"graph_{config.training_version}_{builder.data_mode}_{timestamp}.pt"
        graph_path = os.path.join(config.paths['results'], graph_filename)
        
        os.makedirs(config.paths['results'], exist_ok=True)
        torch.save(graph, graph_path)
        
        # Log graph artifact
        tracker.log_artifact(graph_path, "graphs")
        
        # Run analysis if requested
        if args.analyze:
            print("\nRunning graph analysis...")
            analysis_results = standard_graph_analysis(graph)
            
            # Save analysis results
            analysis_path = graph_path.replace('.pt', '_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Log analysis artifact
            tracker.log_artifact(analysis_path, "analysis")
            
            # Log key analysis metrics
            if 'degree_stats' in analysis_results:
                for key, value in analysis_results['degree_stats'].items():
                    if isinstance(value, (int, float)):
                        tracker.log_metric(f"graph_degree_{key}", value)
            
            print(f"Analysis saved to: {analysis_path}")
        
        print(f"\nGraph creation completed!")
        print(f"Graph saved to: {graph_path}")
        print(f"Nodes: {graph.x.size(0):,}")
        print(f"Edges: {graph.edge_index.size(1):,}")
        print(f"Features: {graph.x.size(1)}")
        print(f"\nMLflow tracking URI: {tracker.experiment_name}")
        print(f"Run ID: {tracker.run_id}")
        
        tracker.end_run()
        return graph_path
        
    except Exception as e:
        print(f"Error during graph creation: {e}")
        import traceback
        traceback.print_exc()
        tracker.end_run()
        raise


if __name__ == "__main__":
    main()
