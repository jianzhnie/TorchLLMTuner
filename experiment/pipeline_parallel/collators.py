"""Experimental pipeline batching reuses independent context-row collation."""

from llmtuner.datasets.collators import ContextBatchCollator as PipelineTextCollator

__all__ = ["PipelineTextCollator"]
