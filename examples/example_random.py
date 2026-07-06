"""
Example: Independent Random Signals

This example demonstrates the rhoSMFDCCA coefficient computed for two
independent random signals. The baseline algorithm should yield a coefficient
fluctuating around zero.

This script uses the pre-generated rand1.csv and rand2.csv datasets.
"""

import sys
from pathlib import Path

# Repository root: resolved dynamically from this script's location
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from smfdcca import mfdcca_coefficient, generate_fibonacci_windows, read_two_separate_csvs
from export import results_to_rows, export_to_csv
from plotting import plot_publication_figure

# Explicit output naming
CSV_NAME = "results_random.csv"
FIGURE_NAME = "rhoSMFDCCA_RANDOM"

def main():
    data_dir = REPO_ROOT / "data"
    file_x = data_dir / "rand1.csv"
    file_y = data_dir / "rand2.csv"
    
    if not file_x.exists() or not file_y.exists():
        print("================================================================")
        print("ERROR: Manuscript datasets missing.")
        print(f"Required files:")
        print(f"  - {file_x}")
        print(f"  - {file_y}")
        print("Please place the rand1.csv and rand2.csv series in the")
        print("'data' directory before running this example.")
        print("================================================================")
        sys.exit(1)
        
    print(f"Reading {file_x.name} and {file_y.name}...")
    x, y, _ = read_two_separate_csvs(str(file_x), str(file_y))
    
    windows = generate_fibonacci_windows(len(x) // 4)
    q_values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    
    print("Computing rhoSMFDCCA...")
    results = mfdcca_coefficient(
        x, y, 
        windows=windows, 
        q_values=q_values,
        pol_ord=1, 
        rev_seg=True, 
        overlap=True
    )
    
    # Format and export
    rows = results_to_rows(results)
    
    # Export CSV
    output_dir = REPO_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    csv_out = output_dir / CSV_NAME
    export_to_csv(rows, str(csv_out))
    
    # Generate Plot
    print("Generating publication figure...")
    fig_dir = REPO_ROOT / "figures"
    plot_publication_figure(str(csv_out), output_dir=str(fig_dir),
                           output_name=FIGURE_NAME)

if __name__ == "__main__":
    main()
