"""Promotion of candidate documents to references during reference runs.

Used by ``VisualTest`` and ``PdfTest`` when ``reference_run`` is active:
instead of failing on a missing or differing reference, the candidate file
is saved as the new reference.
"""

import os
import shutil


def promote_candidate_to_reference(reference_path: str, candidate_path: str) -> str:
    """Copy ``candidate_path`` over ``reference_path`` and return the target path.

    Creates missing parent directories of the reference. The candidate must
    exist as a local file.
    """
    reference_path = os.fspath(reference_path)
    candidate_path = os.fspath(candidate_path)
    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(
            f"Cannot save candidate as reference: '{candidate_path}' does not exist"
        )
    if os.path.abspath(reference_path) == os.path.abspath(candidate_path):
        return reference_path
    parent = os.path.dirname(os.path.abspath(reference_path))
    os.makedirs(parent, exist_ok=True)
    shutil.copyfile(candidate_path, reference_path)
    return reference_path
