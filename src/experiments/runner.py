"""
Experiment runner that executes experiments based on YAML configuration.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import pandas as pd
from ..config.loader import load_config, ExperimentConfig
from ..config.schema import RetrieverConfig
from ..datasets.loader import load_beir_dataset, select_split, as_documents, as_queries
from ..retrievers.factory import create_retriever
from ..eval.evaluator import evaluate_predictions
from ..eval.formatters import get_formatter
from ..utils.io import ensure_dir
from ..utils.logging import get_logger, log_time

_log = get_logger("experiments.runner")


class ExperimentRunner:
    """Runner for executing retrieval experiments from configuration."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.log = get_logger(f"experiment[{config.experiment.get('name', 'unnamed')}]")
    
    def run(self) -> pd.DataFrame:
        """Run the experiment and return results.
        
        Returns:
            DataFrame with results for all retrievers across all datasets
        """
        # Get list of datasets to process
        datasets = self.config.get_datasets()
        
        self.log.info(f"Running experiment on {len(datasets)} dataset(s)")
        
        all_results_rows = []
        
        # Process each dataset
        for dataset_config in datasets:
            dataset_name = dataset_config.name
            self.log.info(f"\n{'#'*80}")
            self.log.info(f"# Processing Dataset: {dataset_name}")
            self.log.info(f"{'#'*80}")
            
            # Load dataset
            dataset_root = Path(dataset_config.root or 
                              f"./data/{dataset_name}/processed/beir")
            
            if not dataset_root.exists():
                self.log.warning(f"⚠️  Dataset root not found: {dataset_root}. Skipping...")
                # Add error rows for this dataset
                for retriever_config in self.config.retrievers:
                    retriever_name = retriever_config.name or retriever_config.type
                    for k in self.config.ks:
                        error_row = {
                            "k": k,
                            "retriever": retriever_name,
                            "retriever_type": retriever_config.type,
                            "dataset": dataset_name,
                            "split": "unknown",
                            "error": f"Dataset root not found: {dataset_root}",
                        }
                        for metric in self.config.metrics:
                            error_row[metric] = 0.0
                        all_results_rows.append(error_row)
                continue
            
            self.log.info(f"Loading dataset from: {dataset_root}")
            try:
                with log_time(self.log, f"Load dataset ({dataset_name})"):
                    corpus, queries, qrels = load_beir_dataset(dataset_root)
            except Exception as e:
                self.log.error(f"❌ Error loading dataset {dataset_name}: {e}", exc_info=True)
                # Add error rows
                for retriever_config in self.config.retrievers:
                    retriever_name = retriever_config.name or retriever_config.type
                    for k in self.config.ks:
                        error_row = {
                            "k": k,
                            "retriever": retriever_name,
                            "retriever_type": retriever_config.type,
                            "dataset": dataset_name,
                            "split": "unknown",
                            "error": str(e),
                        }
                        for metric in self.config.metrics:
                            error_row[metric] = 0.0
                        all_results_rows.append(error_row)
                continue
            
            # Select split
            split = select_split(qrels, tuple(dataset_config.split_preference))
            split_eval = "test" if "test" in set(qrels["split"]) else split
            qrels_eval = qrels[qrels["split"] == split_eval].copy()
            
            self.log.info(f"Using split: {split_eval} ({len(qrels_eval)} qrels)")
            
            # Prepare documents and queries
            docs = as_documents(corpus)
            qids = set(qrels_eval["query_id"].unique().tolist())
            queries_eval = queries[queries["query_id"].isin(qids)]
            qlist = as_queries(queries_eval)
            
            self.log.info(f"Documents: {len(docs)}, Queries: {len(qlist)}")
            
            # Run each retriever on this dataset
            for retriever_config in self.config.retrievers:
                retriever_name = retriever_config.name or retriever_config.type
                self.log.info(f"\n{'='*80}")
                self.log.info(f"Running retriever: {retriever_name} on {dataset_name}")
                self.log.info(f"{'='*80}")
                
                try:
                    # Create retriever from config
                    retriever_dict = retriever_config.model_dump(exclude_none=True)
                    # Remove 'name' from dict as it's not part of retriever config
                    retriever_dict.pop("name", None)
                    retriever = create_retriever(retriever_dict)
                    
                    # Build index
                    with log_time(self.log, f"Build index ({retriever_name} on {dataset_name})"):
                        retriever.build_index(docs)
                    
                    # Retrieve
                    t0 = time.time()
                    preds = retriever.retrieve(qlist, k=max(self.config.ks))
                    t_retrieve = time.time() - t0
                    
                    # Evaluate
                    with log_time(self.log, f"Evaluate ({retriever_name} on {dataset_name})"):
                        metrics_df = evaluate_predictions(
                            preds,
                            qrels_eval,
                            ks=tuple(self.config.ks),
                            metrics=self.config.metrics,
                        )
                    
                    # Add metadata
                    for _, row in metrics_df.iterrows():
                        result_row = row.to_dict()
                        result_row.update({
                            "retriever": retriever_name,
                            "retriever_type": retriever_config.type,
                            "dataset": dataset_name,
                            "split": split_eval,
                            "t_retrieve_sec": round(t_retrieve, 3),
                        })
                        all_results_rows.append(result_row)
                    
                    self.log.info(f"✅ Retriever {retriever_name} on {dataset_name} completed")
                    
                except Exception as e:
                    self.log.error(f"❌ Error running retriever {retriever_name} on {dataset_name}: {e}", exc_info=True)
                    # Add error row
                    for k in self.config.ks:
                        error_row = {
                            "k": k,
                            "retriever": retriever_name,
                            "retriever_type": retriever_config.type,
                            "dataset": dataset_name,
                            "split": split_eval,
                            "error": str(e),
                        }
                        for metric in self.config.metrics:
                            error_row[metric] = 0.0
                        all_results_rows.append(error_row)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results_rows)
        
        # Save results
        if self.config.output_dir:
            self._save_results(results_df)
        
        return results_df
    
    def _save_results(self, results_df: pd.DataFrame):
        """Save results in all requested formats."""
        output_dir = Path(self.config.output_dir)
        ensure_dir(output_dir)
        
        base_name = self.config.experiment.get("name", "experiment")
        
        for fmt_name in self.config.output_formats:
            formatter = get_formatter(fmt_name)
            output_path = output_dir / f"{base_name}.{fmt_name}"
            formatter.format(results_df, output_path=str(output_path))
            self.log.info(f"Saved results to: {output_path}")


def run_experiment(config_path: str | Path) -> pd.DataFrame:
    """Run an experiment from a configuration file.
    
    Args:
        config_path: Path to YAML/JSON configuration file
        
    Returns:
        DataFrame with experiment results
    """
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    return runner.run()

