"""
Script principal para executar experimentos via arquivo de configuração YAML.

Uso:
    python scripts/run_experiment.py --config configs/experiment.yaml
    python scripts/run_experiment.py --config configs/experiment.yaml --override experiment.name=my_exp
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Torna o repo importável
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.runner import run_experiment
from src.config.loader import apply_cli_overrides, load_config
from src.utils.logging import get_logger


def parse_args():
    p = argparse.ArgumentParser(
        description="Run retrieval experiment from YAML/JSON configuration"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file (YAML or JSON)"
    )
    p.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override config values (e.g., --override experiment.name=my_exp --override dataset.name=fiqa)"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (takes precedence over config)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    log = get_logger("run_experiment")
    
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    log.info("="*80)
    log.info("🔬 EXPERIMENT RUNNER")
    log.info("="*80)
    log.info(f"Configuration: {config_path}")
    
    # Load config
    config = load_config(config_path)
    
    # Apply CLI overrides
    if args.override:
        override_dict = {}
        for override in args.override:
            if "=" not in override:
                log.warning(f"Invalid override format (missing =): {override}")
                continue
            key, value = override.split("=", 1)
            override_dict[key] = value
        
        config_dict = config.model_dump()
        config_dict = apply_cli_overrides(config_dict, override_dict)
        config = type(config)(**config_dict)
    
    # Override output dir if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Run experiment
    from src.experiments.runner import ExperimentRunner
    runner = ExperimentRunner(config)
    results = runner.run()
    
    log.info("="*80)
    log.info("✅ EXPERIMENT COMPLETED")
    log.info("="*80)
    
    # Print summary
    if not results.empty:
        print("\n=== Results Summary ===")
        with pd.option_context("display.max_columns", None, "display.width", None):
            print(results.to_string(index=False))


if __name__ == "__main__":
    import pandas as pd
    main()

