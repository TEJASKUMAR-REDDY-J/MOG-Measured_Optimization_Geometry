"""Evaluation. Validation loss is computed on fixed held-out batches
(CharData.eval_batches with data_seed EVAL_STREAM), shared by every run."""

from .trainer import evaluate

__all__ = ["evaluate"]
