"""Data loading and utilities for Amazon ESCI dataset.

Provides ESCIDataLoader for loading/merging parquets, product text expansion,
and train/val/test splitting. Re-exports esci_label2relevance_pos and
get_product_expanded_text for convenience.
"""

from .utils import esci_label2relevance_pos, get_product_expanded_text

__all__ = ["esci_label2relevance_pos", "get_product_expanded_text"]
