"""
Epsilon Robustness Evaluation for rhoSMFDCCA.

This utility numerically validates the effect of the regularization parameter (epsilon)
on the rhoSMFDCCA coefficient without modifying the baseline algorithm.
It automatically executes the epsilon robustness analysis for every manuscript example
available in the repository.
"""

import sys
import time
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smfdcca import mfdcca_coefficient, generate_fibonacci_windows, read_two_separate_csvs
from export import results_to_rows

EPSILONS = [1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6]
BASELINE_EPS = 1e-12

# Representative manuscript fluctuation orders for comparison figures
Q_VALUES_PLOT = [-5.0, -2.0, 0.0, 2.0, 5.0]

EXAMPLES = [
    {
        "name": "RANDOM_SIGNALS",
        "file_x": "rand1.csv",
        "file_y": "rand2.csv",
    },
    {
        "name": "DJI_IXIC",
        "file_x": "DJI_close_log_returns.csv",
        "file_y": "IXIC_close_log_returns.csv",
    },
    {
        "name": "SP_TEMP_RH",
        "file_x": "Temp_Mean_daily_SP.csv",
        "file_y": "Humid_Mean_daily_SP.csv",
    },
    {
        "name": "SP_TMAX_TMIN",
        "file_x": "Temp_Max_daily_SP.csv",
        "file_y": "Temp_Min_daily_SP.csv",
    },
]


def gather_metadata() -> dict:
    """Generates reproducibility metadata describing the execution environment."""
    metadata = {
        "execution_time_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "operating_system": platform.platform(),
    }
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        metadata["git_commit_hash"] = commit
    except Exception:
        metadata["git_commit_hash"] = "Not available or not a git repository"
        
    return metadata


def process_dataset(config: dict, data_dir: Path, out_dir: Path) -> list[dict]:
    """Processes a single dataset pair for the epsilon study."""
    dataset_name = config["name"]
    file_x = data_dir / config["file_x"]
    file_y = data_dir / config["file_y"]
    
    if not file_x.exists() or not file_y.exists():
        print(f"WARNING: Datasets missing for {dataset_name}. Skipping.")
        return []
        
    print(f"\n--- Processing {dataset_name} ---")
    x, y, _ = read_two_separate_csvs(str(file_x), str(file_y))
    
    windows = generate_fibonacci_windows(len(x) // 4)
    q_values = [-5, -2, 0, 2, 5]
    
    complete_rows = []
    summary_rows = []
    global_summary_rows = []
    baseline_map = {}
    
    # Calculate baseline
    print(f"Computing baseline with epsilon = {BASELINE_EPS}...")
    baseline_res = mfdcca_coefficient(x, y, windows=windows, q_values=q_values, pol_ord=1, eps=BASELINE_EPS)
    
    for row in results_to_rows(baseline_res):
        baseline_map[(row['window'], row['q'])] = row['rhoSMFDCCA']
        
    # Main loop over epsilons
    for eps in EPSILONS:
        start_t = time.time()
        res = mfdcca_coefficient(x, y, windows=windows, q_values=q_values, pol_ord=1, eps=eps)
        exec_time = time.time() - start_t
        
        diffs = []
        for row in results_to_rows(res):
            w = row['window']
            q = row['q']
            rho = row['rhoSMFDCCA']
            
            complete_rows.append({
                "window": w,
                "q": q,
                "rho": rho,
                "epsilon": eps
            })
            
            base_rho = baseline_map[(w, q)]
            diffs.append(rho - base_rho)
            
        diffs = np.array(diffs)
        rmse = np.sqrt(np.mean(diffs**2))
        max_diff = np.max(np.abs(diffs))
        mean_diff = np.mean(diffs)
        
        summary_rows.append({
            "epsilon": eps,
            "execution_time": exec_time,
            "rmse": rmse,
            "maximum_difference": max_diff,
            "mean_difference": mean_diff
        })
        
        global_summary_rows.append({
            "dataset": dataset_name,
            "epsilon": eps,
            "execution_time": exec_time,
            "rmse": rmse,
            "maximum_difference": max_diff,
            "mean_difference": mean_diff
        })
        
    # Export CSVs
    dataset_out_dir = out_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    
    df_complete = pd.DataFrame(complete_rows)
    df_summary = pd.DataFrame(summary_rows)
    
    df_complete.to_csv(dataset_out_dir / "complete.csv", index=False)
    df_summary.to_csv(dataset_out_dir / "summary.csv", index=False)
    
    # Generate Figures
    generate_figures(df_complete, df_summary, dataset_out_dir)
    
    return global_summary_rows


def generate_figures(df_complete: pd.DataFrame, df_summary: pd.DataFrame, out_dir: Path):
    """Generates the publication figures for the epsilon study."""
    
    # 1. Summary Figure
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    color = 'tab:red'
    ax1.set_xlabel('Epsilon (log scale)', fontweight='bold', fontsize=16)
    ax1.set_ylabel(r'Maximum $|\Delta\rho|$', color=color, fontweight='bold', fontsize=16)
    ax1.plot(df_summary['epsilon'], df_summary['maximum_difference'], marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('RMSE', color=color, fontweight='bold', fontsize=16)
    ax2.plot(df_summary['epsilon'], df_summary['rmse'], marker='s', color=color, linewidth=2, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.savefig(out_dir / "epsilon_summary.png", bbox_inches='tight', dpi=600)
    plt.savefig(out_dir / "epsilon_summary.pdf", bbox_inches='tight')
    plt.savefig(out_dir / "epsilon_summary.svg", bbox_inches='tight')
    plt.close()
    
    # 2. Comparison Figures
    colors_eps = plt.cm.viridis(np.linspace(0, 1, len(EPSILONS)))
    for target_q in Q_VALUES_PLOT:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for eps, c in zip(EPSILONS, colors_eps):
            df_eps = df_complete[
                np.isclose(df_complete['epsilon'], eps) &
                np.isclose(df_complete['q'], target_q)
            ]
            ax.plot(df_eps['window'], df_eps['rho'], label=f'eps={eps:g}',
                    color=c, marker='.', linestyle='None', alpha=0.7)
            
        ax.set_xscale('log')
        ax.set_xlabel(r'$n$', fontsize=16, fontweight='bold')
        ax.set_ylabel(r'$\rho$', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(-1.1, 1.1)
        
        legend = ax.legend(title="Epsilon", bbox_to_anchor=(1.02, 1), loc='upper left',
                           fontsize=14, title_fontsize=16)
        legend.get_title().set_fontweight('bold')
        
        q_tag = f"q_{int(target_q)}" if target_q >= 0 else f"q_neg{int(abs(target_q))}"
        plt.tight_layout()
        plt.savefig(out_dir / f"comparison_{q_tag}.png", bbox_inches='tight', dpi=600)
        plt.savefig(out_dir / f"comparison_{q_tag}.pdf", bbox_inches='tight')
        plt.savefig(out_dir / f"comparison_{q_tag}.svg", bbox_inches='tight')
        plt.close()


def run_study():
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    out_dir = root_dir / "results" / "epsilon_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_global_rows = []
    
    # Run all examples
    for config in EXAMPLES:
        rows = process_dataset(config, data_dir, out_dir)
        all_global_rows.extend(rows)
        
    # Global Summary
    if all_global_rows:
        df_global = pd.DataFrame(all_global_rows)
        df_global.to_csv(out_dir / "epsilon_summary_all_examples.csv", index=False)
        print("\nSaved global summary: epsilon_summary_all_examples.csv")
        
    # Metadata
    meta = gather_metadata()
    with open(out_dir / "epsilon_study_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
    print("Saved global metadata: epsilon_study_metadata.json")


if __name__ == "__main__":
    run_study()
