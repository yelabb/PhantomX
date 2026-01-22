"""
PhantomX CLI

Unified command-line interface for dataset management and experiments.

Usage:
    python -m phantomx datasets list
    python -m phantomx datasets download mc_maze
    python -m phantomx datasets info mc_rtt
    
    python -m phantomx run baseline --dataset mc_maze
    python -m phantomx run experiment --config distillation.yaml
"""

import argparse
import sys
from pathlib import Path


def cmd_datasets(args):
    """Handle dataset commands."""
    from phantomx.datasets import get_dataset, list_datasets, DATASET_REGISTRY
    from phantomx.datasets.registry import _import_all_datasets
    from phantomx.datasets.base import get_cache_dir
    from pathlib import Path
    import shutil
    
    _import_all_datasets()
    
    if args.action == "list":
        list_datasets(verbose=True)
        
    elif args.action == "download":
        if not args.name:
            print("Error: dataset name required. Use: phantomx datasets download <name>")
            sys.exit(1)
        
        if args.name == "all":
            for name in DATASET_REGISTRY:
                ds = get_dataset(name)
                ds.download(force=args.force)
        else:
            ds = get_dataset(args.name)
            ds.download(force=args.force)
            
    elif args.action == "info":
        if not args.name:
            print("Error: dataset name required")
            sys.exit(1)
        ds = get_dataset(args.name)
        print(ds.summary())
        
    elif args.action == "cache":
        cache_dir = get_cache_dir()
        print(f"Cache directory: {cache_dir}")
        
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            print(f"Total size: {total_size / 1e6:.1f} MB")
            print("\nCached datasets:")
            for d in cache_dir.iterdir():
                if d.is_dir():
                    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    print(f"  {d.name}: {size / 1e6:.1f} MB")
        else:
            print("  (empty)")
    
    elif args.action == "link":
        # Link a local file to the cache
        if not args.name or not args.path:
            print("Error: requires dataset name and --path")
            print("Usage: phantomx datasets link mc_maze --path data/mc_maze.nwb")
            sys.exit(1)
        
        local_path = Path(args.path).resolve()
        if not local_path.exists():
            print(f"Error: file not found: {local_path}")
            sys.exit(1)
        
        ds = get_dataset(args.name)
        cache_path = ds.cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        if cache_path.exists():
            cache_path.unlink()
        
        # On Windows, copy instead of symlink (requires admin)
        if sys.platform == "win32":
            print(f"Copying {local_path} -> {cache_path}")
            shutil.copy2(local_path, cache_path)
        else:
            print(f"Symlinking {local_path} -> {cache_path}")
            cache_path.symlink_to(local_path)
        
        print(f"✓ {args.name} linked successfully")


def cmd_run(args):
    """Handle experiment run commands."""
    from phantomx.config import ExperimentConfig, get_preset, list_presets, PRESETS
    from phantomx.runner import ExperimentRunner
    from pathlib import Path
    
    # Load configuration
    if args.config:
        # Load from YAML file
        config_path = Path(args.config)
        if not config_path.exists():
            # Try configs/ directory
            config_path = Path("configs") / args.config
            if not config_path.suffix:
                config_path = config_path.with_suffix(".yaml")
        
        if not config_path.exists():
            print(f"Error: config file not found: {args.config}")
            sys.exit(1)
        
        print(f"Loading config from: {config_path}")
        config = ExperimentConfig.from_yaml(str(config_path))
    
    elif args.preset:
        # Use preset
        if args.preset == "list":
            list_presets()
            return
        
        try:
            config = get_preset(args.preset)
        except KeyError as e:
            print(f"Error: {e}")
            print("Available presets:")
            list_presets()
            sys.exit(1)
    
    else:
        # Default: list available options
        print("Usage:")
        print("  phantomx run --config configs/lstm_baseline.yaml")
        print("  phantomx run --preset lstm_baseline")
        print("\nAvailable configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for f in sorted(configs_dir.glob("*.yaml")):
                print(f"  {f.stem}")
        print("\nAvailable presets:")
        list_presets()
        return
    
    # Override dataset if specified
    if args.dataset:
        config.dataset.name = args.dataset
    
    # Override seeds if specified
    if args.seeds:
        config.training.seeds = [int(s) for s in args.seeds.split(",")]
    
    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test R² = {results['test_r2_mean']:.4f} ± {results['test_r2_std']:.4f}")
    print(f"Range: [{results['test_r2_min']:.4f}, {results['test_r2_max']:.4f}]")


def cmd_models(args):
    """List available models."""
    from phantomx.models import list_models
    list_models()


def cmd_compare(args):
    """Compare results across datasets/experiments."""
    print(f"Comparing: {args.experiments}")
    print(f"Datasets: {args.datasets}")
    
    # TODO: Implement comparison
    print("\n⚠️  Comparison not yet implemented.")


def main():
    parser = argparse.ArgumentParser(
        prog="phantomx",
        description="PhantomX: Neural tokenization for BCIs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # === datasets ===
    ds_parser = subparsers.add_parser("datasets", help="Manage datasets")
    ds_parser.add_argument("action", choices=["list", "download", "info", "cache", "link"],
                          help="Action to perform")
    ds_parser.add_argument("name", nargs="?", help="Dataset name (or 'all')")
    ds_parser.add_argument("--force", "-f", action="store_true", 
                          help="Force re-download")
    ds_parser.add_argument("--path", "-p", help="Local file path (for 'link' action)")
    ds_parser.set_defaults(func=cmd_datasets)
    
    # === run ===
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--config", "-c", help="Config YAML file path")
    run_parser.add_argument("--preset", "-p", help="Use a preset config (or 'list')")
    run_parser.add_argument("--dataset", "-d", help="Override dataset")
    run_parser.add_argument("--seeds", "-s", help="Override seeds (comma-separated)")
    run_parser.set_defaults(func=cmd_run)
    
    # === models ===
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.set_defaults(func=cmd_models)
    
    # === compare ===
    cmp_parser = subparsers.add_parser("compare", help="Compare results")
    cmp_parser.add_argument("experiments", nargs="+", help="Experiments to compare")
    cmp_parser.add_argument("--datasets", "-d", nargs="+", default=["mc_maze"])
    cmp_parser.set_defaults(func=cmd_compare)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
