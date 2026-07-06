"""
Publication-quality Plotting Utilities for rhoSMFDCCA.

This module reads from standardized CSV files (window,q,rho) and generates
manuscript-ready scatter plots without significance styling.

This plotting style is the official visualization standard for the rhoSMFDCCA
repository and should be reused consistently across all manuscript examples.
"""

import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Selected q values for traditional manuscript representation
Q_TO_PLOT = [-5.0, -2.0, 0.0, 2.0, 5.0]
MARKERS = ['o', 's', '^', 'D', 'v']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def read_data(csv_path: str) -> pd.DataFrame:
    """Reads and validates the CSV, ensuring it conforms to standard publication format."""
    df = pd.read_csv(csv_path, comment='#')
    required_cols = {'window', 'q', 'rho'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV '{csv_path}' missing required columns: {required_cols}")
    return df

def plot_publication_figure(csv_path: str, output_dir: str = "figures/",
                            output_name: str | None = None):
    """Generates a standalone publication figure for a single CSV.

    Args:
        csv_path: Path to the standardized CSV file (window,q,rho).
        output_dir: Directory where figures will be saved.
        output_name: Explicit output file basename (without extension).
                     If None, derived from the CSV filename.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        df = read_data(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if output_name is None:
        stem = Path(csv_path).stem
        output_name = f"rhoSMFDCCA_{stem.replace('results_', '').upper()}"

    out_prefix = Path(output_dir) / output_name
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    for q_val, marker, color in zip(Q_TO_PLOT, MARKERS, COLORS):
        df_q = df[np.isclose(df['q'], q_val)].sort_values('window')
        if df_q.empty:
            continue
            
        windows = df_q['window'].values
        rhos = df_q['rho'].values
        
        # Plot standard filled markers (no significance formatting)
        ax.plot(windows, rhos, marker=marker, linestyle='None',
                markerfacecolor=color, markeredgecolor=color, markersize=10)
                    
    ax.set_xscale('log')
    ax.set_xlabel(r'$n$', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$\rho$', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim(-1.1, 1.1)
    
    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------
    
    # q-values legend
    q_handles = []
    for q_val, marker, color in zip(Q_TO_PLOT, MARKERS, COLORS):
        q_handles.append(Line2D([0], [0], marker=marker, color='w', label=f'q = {q_val:g}',
                                markerfacecolor=color, markeredgecolor=color, markersize=10))
                                
    # Placed completely outside the axes, upper right
    legend_q = ax.legend(handles=q_handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
                         fontsize=14, framealpha=0.9, title="Fluctuation order", title_fontsize=16)
    legend_q.get_title().set_fontweight('bold')
    
    plt.savefig(f"{out_prefix}.png", bbox_inches='tight', dpi=600)
    plt.savefig(f"{out_prefix}.pdf", bbox_inches='tight')
    plt.savefig(f"{out_prefix}.svg", bbox_inches='tight')
    plt.close()
    
    print(f"Saved {out_prefix}.png, .pdf, and .svg")

def plot_publication_multipanel(csv_paths: list[str], output_dir: str = "figures/",
                                output_name: str = "rhoSMFDCCA_MULTIPLOT"):
    """Generates a 4-panel (2x2) publication figure from a list of 4 CSV files.
    
    Args:
        csv_paths: List of 4 standardized CSV file paths (window,q,rho).
        output_dir: Directory where figures will be saved.
        output_name: Explicit output file basename (without extension).
    """
    if len(csv_paths) != 4:
        raise ValueError("plot_publication_multipanel expects exactly 4 CSV paths for a 2x2 layout.")
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_prefix = Path(output_dir) / output_name
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    
    for i, csv_path in enumerate(csv_paths):
        ax = axes[i]
        
        try:
            df = read_data(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
            
        for q_val, marker, color in zip(Q_TO_PLOT, MARKERS, COLORS):
            df_q = df[np.isclose(df['q'], q_val)].sort_values('window')
            if df_q.empty:
                continue
                
            windows = df_q['window'].values
            rhos = df_q['rho'].values
            
            # Plot standard filled markers
            ax.plot(windows, rhos, marker=marker, linestyle='None',
                    markerfacecolor=color, markeredgecolor=color, markersize=10)
                    
        ax.set_xscale('log')
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Subplot labels
        ax.text(0.05, 0.95, panel_labels[i], transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top', ha='left')
                
    # Shared labels
    fig.supxlabel(r'$n$', fontsize=16, fontweight='bold', y=0.02)
    fig.supylabel(r'$\rho$', fontsize=16, fontweight='bold', x=0.02)
    
    # ---------------------------------------------------------
    # Shared Legend
    # ---------------------------------------------------------
    q_handles = []
    for q_val, marker, color in zip(Q_TO_PLOT, MARKERS, COLORS):
        q_handles.append(Line2D([0], [0], marker=marker, color='w', label=f'q = {q_val:g}',
                                markerfacecolor=color, markeredgecolor=color, markersize=10))
                                
    # Placed completely outside the axes, right side of the figure
    legend_q = fig.legend(handles=q_handles, loc='center right', bbox_to_anchor=(1.12, 0.5), 
                          fontsize=14, framealpha=0.9, title="Fluctuation order", title_fontsize=16)
    legend_q.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    # Adjust layout to make room for shared labels and legend
    fig.subplots_adjust(bottom=0.1, left=0.1, right=0.9)
    
    plt.savefig(f"{out_prefix}.png", bbox_inches='tight', dpi=600)
    plt.savefig(f"{out_prefix}.pdf", bbox_inches='tight')
    plt.savefig(f"{out_prefix}.svg", bbox_inches='tight')
    plt.close()
    
    print(f"Saved {out_prefix}.png, .pdf, and .svg")


def plot_publication_twopanel_vertical(csv_paths: list[str], output_dir: str = "figures/",
                                       output_name: str = "rhoSMFDCCA_TWOPANEL"):
    """Generates a 2-panel (2x1 vertical) publication figure from a list of 2 CSV files.
    
    Args:
        csv_paths: List of 2 standardized CSV file paths (window,q,rho).
        output_dir: Directory where figures will be saved.
        output_name: Explicit output file basename (without extension).
    """
    if len(csv_paths) != 2:
        raise ValueError("plot_publication_twopanel_vertical expects exactly 2 CSV paths for a 2x1 layout.")
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_prefix = Path(output_dir) / output_name
    
    # Portrait orientation: 8x10
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), dpi=300, sharex=True, sharey=True)
    panel_labels = ['(a)', '(b)']
    
    for i, csv_path in enumerate(csv_paths):
        ax = axes[i]
        
        try:
            df = read_data(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
            
        for q_val, marker, color in zip(Q_TO_PLOT, MARKERS, COLORS):
            df_q = df[np.isclose(df['q'], q_val)].sort_values('window')
            if df_q.empty:
                continue
                
            windows = df_q['window'].values
            rhos = df_q['rho'].values
            
            # Plot standard filled markers
            ax.plot(windows, rhos, marker=marker, linestyle='None',
                    markerfacecolor=color, markeredgecolor=color, markersize=10)
                    
        ax.set_xscale('log')
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Subplot labels
        ax.text(0.05, 0.95, panel_labels[i], transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top', ha='left')
                
    # Shared labels
    fig.supxlabel(r'$n$', fontsize=16, fontweight='bold', y=0.02)
    fig.supylabel(r'$\rho$', fontsize=16, fontweight='bold', x=0.02)
    
    # ---------------------------------------------------------
    # Shared Legend
    # ---------------------------------------------------------
    q_handles = []
    for q_val, marker, color in zip(Q_TO_PLOT, MARKERS, COLORS):
        q_handles.append(Line2D([0], [0], marker=marker, color='w', label=f'q = {q_val:g}',
                                markerfacecolor=color, markeredgecolor=color, markersize=10))
                                
    # Placed completely outside the axes, right side of the figure
    legend_q = fig.legend(handles=q_handles, loc='center right', bbox_to_anchor=(1.25, 0.5), 
                          fontsize=14, framealpha=0.9, title="Fluctuation order", title_fontsize=16)
    legend_q.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    # Adjust layout to make room for shared labels and legend
    fig.subplots_adjust(bottom=0.1, left=0.15, right=0.85)
    
    plt.savefig(f"{out_prefix}.png", bbox_inches='tight', dpi=600)
    plt.savefig(f"{out_prefix}.pdf", bbox_inches='tight')
    plt.savefig(f"{out_prefix}.svg", bbox_inches='tight')
    plt.close()
    
    print(f"Saved {out_prefix}.png, .pdf, and .svg")


def main():
    parser = argparse.ArgumentParser(description="Plot publication-ready 2D SMFDCCA points.")
    parser.add_argument("csv_files", nargs='+', help="Path to CSV results files.")
    parser.add_argument("--outdir", default="figures/", help="Output directory for figures.")
    args = parser.parse_args()
    
    print(f"Processing {len(args.csv_files)} files...")
    for f in args.csv_files:
        plot_publication_figure(f, output_dir=args.outdir)
        
    print("All figures successfully generated.")

if __name__ == "__main__":
    main()
