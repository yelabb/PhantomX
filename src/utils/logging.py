"""
Experiment Logging Utilities

Integrates with Weights & Biases (WandB) for automatic experiment tracking.
Replaces the manual README tables with auto-generated dashboards.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class ExperimentLogger:
    """
    Unified experiment logger supporting WandB and local logging.
    
    Features:
    - Automatic config serialization
    - Git hash tracking for reproducibility
    - Metric logging with step tracking
    - Artifact management (models, configs)
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        use_wandb: bool = True,
        project: str = "PhantomX",
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.project = project
        self.entity = entity
        self.output_dir = Path(output_dir) if output_dir else Path("logs")
        self.run = None
        self.step = 0
        
        # Prepare tags
        self.tags = tags or []
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'name'):
            self.tags.append(cfg.model.name)
        if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'name'):
            self.tags.append(cfg.dataset.name)
        
        self.notes = notes
        if hasattr(cfg, 'notes'):
            self.notes = cfg.notes
    
    def init(self) -> "ExperimentLogger":
        """Initialize the logger."""
        if self.use_wandb:
            self._init_wandb()
        else:
            self._init_local()
        return self
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=getattr(self.cfg, 'experiment_name', None),
            config=config_dict,
            tags=self.tags,
            notes=self.notes,
            dir=str(self.output_dir),
            reinit=True,
        )
        
        # Log git hash if available
        from .seeding import get_git_hash
        git_hash = get_git_hash()
        if git_hash:
            wandb.config.update({"git_hash": git_hash}, allow_val_change=True)
        
        print(f"📊 WandB run initialized: {self.run.url}")
    
    def _init_local(self):
        """Initialize local file logging."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "metrics.jsonl"
        
        # Save config
        config_file = self.output_dir / "config.yaml"
        OmegaConf.save(self.cfg, config_file)
        
        print(f"📊 Local logging to: {self.output_dir}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if step is not None:
            self.step = step
        else:
            self.step += 1
        
        if self.use_wandb and self.run:
            wandb.log(metrics, step=self.step)
        else:
            # Local logging
            log_entry = {"step": self.step, **metrics, "timestamp": datetime.now().isoformat()}
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary metrics."""
        if self.use_wandb and self.run:
            for key, value in summary.items():
                wandb.run.summary[key] = value
        else:
            summary_file = self.output_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
    
    def save_model(self, model_path: Path, name: str = "model"):
        """Save model as artifact."""
        if self.use_wandb and self.run:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(str(model_path))
            self.run.log_artifact(artifact)
        # Local: model already saved by trainer
    
    def finish(self):
        """Finish logging."""
        if self.use_wandb and self.run:
            wandb.finish()
            print("📊 WandB run finished")


# Module-level convenience functions
_logger: Optional[ExperimentLogger] = None


def init_logger(
    cfg: DictConfig,
    use_wandb: bool = True,
    **kwargs
) -> ExperimentLogger:
    """Initialize the global experiment logger."""
    global _logger
    
    # Get settings from config if available
    if hasattr(cfg, 'logging'):
        use_wandb = cfg.logging.get('use_wandb', use_wandb)
        kwargs.setdefault('project', cfg.logging.get('wandb_project', 'PhantomX'))
        kwargs.setdefault('entity', cfg.logging.get('wandb_entity', None))
    
    if hasattr(cfg, 'output_dir'):
        kwargs.setdefault('output_dir', cfg.output_dir)
    
    _logger = ExperimentLogger(cfg, use_wandb=use_wandb, **kwargs)
    return _logger.init()


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics to the global logger."""
    if _logger:
        _logger.log(metrics, step)


def finish_logger():
    """Finish the global logger."""
    global _logger
    if _logger:
        _logger.finish()
        _logger = None
