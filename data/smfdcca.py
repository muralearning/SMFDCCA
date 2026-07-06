"""Signed Multifractal Detrended Cross-Correlation Coefficient (ρ_SMFDCCA).

This module follows the MFDFA/MFDCCA workflow:

1. build integrated profiles from two time series,
2. split the profiles into windows,
3. detrend each window with a polynomial,
4. compute local detrended variances and covariance,
5. combine the local quantities into a q-dependent coefficient.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# Activates Colab's interactive formatting tables automatically
try:
    from google.colab import data_table
    HAS_COLAB = True
except ImportError:
    HAS_COLAB = False


@dataclass(frozen=True)
class MFDCCAResult:
    """Coefficient values for one window size."""

    window: int
    q: np.ndarray
    rhoSMFDCCA: np.ndarray
    n_segments: int


def to_profile(series: Iterable[float], demean: bool = True) -> np.ndarray:
    """Return the integrated profile used by DFA/MFDFA/MFDCCA."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if demean:
        values = values - np.nanmean(values)
    return np.cumsum(values)


def _segment_starts(length: int, window: int, rev_seg: bool, overlap: bool) -> list[int]:
    if window < 3:
        raise ValueError("window must be at least 3")
    if window > length:
        raise ValueError("window cannot be larger than the series length")

    if overlap:
        return list(range(0, length - window + 1))

    starts = list(range(0, length - window + 1, window))
    if rev_seg:
        reverse_starts = list(range(length - window, -1, -window))
        starts.extend(reverse_starts)
    return sorted(set(starts))


def _local_detrended_stats(
    x_profile: np.ndarray,
    y_profile: np.ndarray,
    window: int,
    pol_ord: int,
    rev_seg: bool,
    overlap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = _segment_starts(len(x_profile), window, rev_seg=rev_seg, overlap=overlap)
    t = np.arange(window, dtype=float)

    var_x: list[float] = []
    var_y: list[float] = []
    cov_xy: list[float] = []

    for start in starts:
        stop = start + window
        x_seg = x_profile[start:stop]
        y_seg = y_profile[start:stop]

        x_trend = np.polyval(np.polyfit(t, x_seg, pol_ord), t)
        y_trend = np.polyval(np.polyfit(t, y_seg, pol_ord), t)
        x_res = x_seg - x_trend
        y_res = y_seg - y_trend

        var_x.append(float(np.mean(x_res * x_res)))
        var_y.append(float(np.mean(y_res * y_res)))
        cov_xy.append(float(np.mean(x_res * y_res)))

    return np.asarray(var_x), np.asarray(var_y), np.asarray(cov_xy)


def _weighted_signed_rho(
    var_x: np.ndarray,
    var_y: np.ndarray,
    cov_xy: np.ndarray,
    q: float,
    eps: float,
) -> float:
    amplitude = np.sqrt(np.maximum(var_x, eps) * np.maximum(var_y, eps))
    local_r = cov_xy / amplitude
    local_r = np.clip(local_r, -1.0, 1.0)

    if np.isclose(q, 0.0):
        weights = np.ones_like(amplitude)
    else:
        log_weights = 0.5 * q * np.log(np.maximum(amplitude, eps))
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

    return float(np.sum(weights * local_r) / np.sum(weights))


def mfdcca_coefficient(
    x: Iterable[float],
    y: Iterable[float],
    windows: Iterable[int],
    q_values: Iterable[float],
    pol_ord: int = 1,
    rev_seg: bool = True,
    overlap: bool = True,
    already_profiled: bool = False,
    eps: float = 1e-12,
) -> list[MFDCCAResult]:
    """Compute signed MFDCCA coefficients for several windows and q values."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape/length")
    if x_arr.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if pol_ord < 0:
        raise ValueError("pol_ord must be non-negative")

    x_profile = x_arr if already_profiled else to_profile(x_arr)
    y_profile = y_arr if already_profiled else to_profile(y_arr)
    q_arr = np.asarray(list(q_values), dtype=float)

    results: list[MFDCCAResult] = []
    for window in windows:
        var_x, var_y, cov_xy = _local_detrended_stats(
            x_profile,
            y_profile,
            int(window),
            pol_ord=pol_ord,
            rev_seg=rev_seg,
            overlap=overlap,
        )

        rhoSMFDCCA = np.asarray(
            [_weighted_signed_rho(var_x, var_y, cov_xy, q, eps) for q in q_arr]
        )
        results.append(
            MFDCCAResult(
                window=int(window),
                q=q_arr.copy(),
                rhoSMFDCCA=rhoSMFDCCA,
                n_segments=len(var_x),
            )
        )

    return results


def generate_fibonacci_windows(max_limit: int) -> list[int]:
    """Generates Fibonacci window sizes starting from 13 up to max_limit (N/4).

    The Fibonacci sequence is adopted as the window-size grid following the
    baseline manuscript convention.  The sequence begins at 13 because
    smaller Fibonacci numbers (1, 2, 3, 5, 8) do not provide enough data
    points within each segment for reliable polynomial detrending and
    variance estimation, which are the core operations of the MFDCCA
    workflow.  Starting at 13 ensures that every window contains a
    statistically meaningful number of observations while preserving the
    logarithmically-spaced nature of the Fibonacci progression.

    Args:
        max_limit: Upper bound for window sizes, typically N/4 where N is
                   the length of the time series.

    Returns:
        Sorted list of Fibonacci window sizes in [13, max_limit].
    """
    if max_limit < 13:
        return []

    windows = [13]
    # Set initialization values corresponding to the 13 and 21 steps
    a, b = 13, 21
    while b <= max_limit:
        windows.append(b)
        a, b = b, a + b
    return windows


def read_two_separate_csvs(
    path_x: str,
    path_y: str,
    x_col: str | int = 0,
    y_col: str | int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    """Reads two separate CSV files, extracts one numeric column from each,

    and handles files seamlessly whether they have headers or raw text.
    """
    try:
        df_x = pd.read_csv(path_x)
        if pd.to_numeric(df_x.columns, errors='coerce').notna().all():
            df_x = pd.read_csv(path_x, header=None)
    except Exception:
        df_x = pd.read_csv(path_x, header=None)

    try:
        df_y = pd.read_csv(path_y)
        if pd.to_numeric(df_y.columns, errors='coerce').notna().all():
            df_y = pd.read_csv(path_y, header=None)
    except Exception:
        df_y = pd.read_csv(path_y, header=None)

    series_x = df_x.iloc[:, x_col] if isinstance(x_col, int) else df_x[x_col]
    series_y = df_y.iloc[:, y_col] if isinstance(y_col, int) else df_y[y_col]

    x_name = series_x.name if (series_x.name and not isinstance(series_x.name, int)) else f"file_x_col_{x_col}"
    y_name = series_y.name if (series_y.name and not isinstance(series_y.name, int)) else f"file_y_col_{y_col}"

    x_numeric = pd.to_numeric(series_x, errors="coerce").to_numpy()
    y_numeric = pd.to_numeric(series_y, errors="coerce").to_numpy()

    valid_mask_x = np.isfinite(x_numeric)
    valid_mask_y = np.isfinite(y_numeric)

    x_clean = x_numeric[valid_mask_x]
    y_clean = y_numeric[valid_mask_y]

    min_len = min(len(x_clean), len(y_clean))
    if min_len < 3:
        raise ValueError("Datasets must have at least 3 matching observations.")

    return x_clean[:min_len], y_clean[:min_len], (str(x_name), str(y_name))


def _parse_number_list(text: str, cast: type = float) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def calculate_mfdcca_from_two_files(
    csv_path_x: str | None = None,
    csv_path_y: str | None = None,
    x_col: str | int = 0,
    y_col: str | int = 0,
    windows: Iterable[int] | None = None,
    q_values: Iterable[float] = (-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    pol_ord: int = 1,
    overlap: bool = True,
    rev_seg: bool = True,
    already_profiled: bool = False,
    eps: float = 1e-12,
    output: str | None = None,
    print_table: bool = True,
) -> pd.DataFrame | list[dict[str, float | int]]:
    """Notebook-friendly pipeline engineered to cross-analyze two separate files."""
    if csv_path_x is None or csv_path_y is None:
        print("Missing file paths. Running default synthetic demonstration setup...")
        rng = np.random.default_rng(7)
        n = 5000
        x = rng.normal(size=n)
        y = -0.65 * x + 0.76 * rng.normal(size=n)
        x_name, y_name = "demo_x", "demo_y"
    else:
        x, y, (x_name, y_name) = read_two_separate_csvs(
            path_x=csv_path_x,
            path_y=csv_path_y,
            x_col=x_col,
            y_col=y_col
        )

    if windows is None:
        max_division_limit = len(x) // 4
        windows = generate_fibonacci_windows(max_division_limit)
        print(f"Generated Fibonacci Windows (Starting from 13; Max N/4 = {max_division_limit}): {windows}")

    results = mfdcca_coefficient(
        x, y, windows, q_values,
        pol_ord=pol_ord, rev_seg=rev_seg, overlap=overlap,
        already_profiled=already_profiled, eps=eps,
    )
    from export import results_to_rows
    rows = results_to_rows(results)

    if output:
        pd.DataFrame(rows).to_csv(output, index=False)

    if print_table:
        print(f"Analysis Complete: Series X={x_name}, Series Y={y_name} | Common Data Points={len(x)}")
        if HAS_COLAB:
            data_table.enable_dataframe_formatter()
            return pd.DataFrame(rows)
        else:
            print("window,n_segments,q,rhoSMFDCCA")
            for row in rows:
                print(f"{row['window']},{row['n_segments']},{row['q']:g},{row['rhoSMFDCCA']:.6f}")

    return rows


# --- Core Execution Router ---
if __name__ == "__main__":
    file_x = "Log_return_CC_USD.csv"
    file_y = "Log_return_LCC_USD.csv"

    if os.path.exists(file_x) and os.path.exists(file_y):
        df_results = calculate_mfdcca_from_two_files(
            csv_path_x=file_x,
            csv_path_y=file_y,
            x_col=0,
            y_col=0,
            windows=None,
            pol_ord=1
        )
    else:
        print(f"⚠️ Could not locate '{file_x}' or '{file_y}' in workspace root. Running demo...")
        df_results = calculate_mfdcca_from_two_files(
            csv_path_x=None,
            csv_path_y=None,
            windows=None,
            pol_ord=1
        )

    display(df_results)
