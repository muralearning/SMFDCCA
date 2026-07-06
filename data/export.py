"""
CSV export utilities for rhoSMFDCCA.

This module provides independent export utilities that format algorithmic outputs
into the standard publication artifact format:

window,q,rho

It also contains the results_to_rows() serialization function, which converts
the list of MFDCCAResult dataclass instances returned by the algorithm into
a flat list of dictionaries suitable for tabular export.
"""

from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING, List, Dict, Union

if TYPE_CHECKING:
    from smfdcca import MFDCCAResult


def results_to_rows(results: list[MFDCCAResult]) -> list[dict[str, float | int]]:
    """Convert a list of MFDCCAResult objects into a flat list of row dictionaries.

    Each dictionary contains the keys 'window', 'n_segments', 'q', and 'rhoSMFDCCA'.
    This is a serialization utility and is not part of the mathematical algorithm.

    Args:
        results: List of MFDCCAResult instances as returned by mfdcca_coefficient().

    Returns:
        A flat list of dictionaries, one per (window, q) combination.
    """
    rows: list[dict[str, float | int]] = []
    for result in results:
        for q, signed in zip(result.q, result.rhoSMFDCCA):
            rows.append(
                {
                    "window": result.window,
                    "n_segments": result.n_segments,
                    "q": float(q),
                    "rhoSMFDCCA": float(signed),
                }
            )
    return rows


def export_to_csv(results: List[Dict[str, Union[float, int]]], output_path: str):
    """
    Exports the algorithmic results to a standardized CSV format.
    
    The standard exported format must be exactly three columns:
    window,q,rho
    
    No additional statistics, no significance information, no implementation metadata.
    
    Args:
        results: List of dictionaries (e.g., from results_to_rows).
                 Expected to contain 'window', 'q', and 'rhoSMFDCCA' keys.
        output_path: Path to the output CSV file.
    """
    df = pd.DataFrame(results)
    
    if df.empty:
        df = pd.DataFrame(columns=['window', 'q', 'rho'])
        df.to_csv(output_path, index=False)
        return

    # Ensure required columns are present from the core algorithm
    required_keys = {'window', 'q', 'rhoSMFDCCA'}
    if not required_keys.issubset(df.columns):
        raise ValueError(f"Results are missing required keys: {required_keys}")
    
    # Select and rename columns to match standard publication format exactly
    export_df = df[['window', 'q', 'rhoSMFDCCA']].copy()
    export_df.rename(columns={'rhoSMFDCCA': 'rho'}, inplace=True)
    
    # Sort for consistent output
    export_df.sort_values(by=['q', 'window'], inplace=True)
    
    # Export to CSV without index
    export_df.to_csv(output_path, index=False)
    print(f"Exported standard publication CSV to: {output_path}")
