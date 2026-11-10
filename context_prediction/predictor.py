"""Synthetic visible-context retrieval; no target observation is accepted."""
import numpy as np


def retrieve(visible_key, references, k=1):
    """Return nearest reference records, with stable reference-id tie breaking.

    This E1 implementation consumes preconstructed visible vectors only; it is
    not the future image encoder or evidence that pixel-level masking works.
    """
    key = np.asarray(visible_key, dtype=float)
    if key.ndim != 1 or not np.isfinite(key).all() or k < 1:
        raise ValueError('expected a finite visible vector and positive k')
    ranked = []
    for ref in references:
        vector = np.asarray(ref['visible_key'], dtype=float)
        if vector.shape != key.shape or not np.isfinite(vector).all():
            raise ValueError('reference visible vector is invalid')
        ranked.append((float(np.linalg.norm(key - vector)), str(ref['id']), ref))
    return [item[2] for item in sorted(ranked, key=lambda item: item[:2])[:k]]
