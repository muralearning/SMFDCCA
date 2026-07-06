"""
Example: Weather Datasets

Reproduces the manuscript example analyzing meteorological variables 
from São Paulo, Brazil. This script evaluates two complementary relationships:

(a) Mean Temperature × Mean Relative Humidity
(b) Maximum Temperature × Minimum Temperature
"""

import sys
from pathlib import Path

# Repository root: resolved dynamically from this script's location
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from smfdcca import mfdcca_coefficient, generate_fibonacci_windows, read_two_separate_csvs
from export import results_to_rows, export_to_csv
from plotting import plot_publication_twopanel_vertical

FIGURE_NAME = "rhoSMFDCCA_WEATHER_SP"

def main():
    data_dir = REPO_ROOT / "data"
    output_dir = REPO_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    
    analyses = [
        {
            "panel": "a",
            "desc": "Mean Temperature x Mean Relative Humidity (SP)",
            "file_x": "Temp_Mean_daily_SP.csv",
            "file_y": "Humid_Mean_daily_SP.csv",
            "csv_out": "results_sp_temp_humidity.csv"
        },
        {
            "panel": "b",
            "desc": "Maximum Temperature x Minimum Temperature (SP)",
            "file_x": "Temp_Max_daily_SP.csv",
            "file_y": "Temp_Min_daily_SP.csv",
            "csv_out": "results_sp_maxmin_temperature.csv"
        }
    ]
    
    csv_paths = []
    
    for analysis in analyses:
        file_x = data_dir / analysis["file_x"]
        file_y = data_dir / analysis["file_y"]
        
        if not file_x.exists() or not file_y.exists():
            print("================================================================")
            print("ERROR: Manuscript datasets missing.")
            print(f"Required files:")
            print(f"  - {file_x}")
            print(f"  - {file_y}")
            print("Please place the required CSVs in the 'data' directory.")
            print("================================================================")
            sys.exit(1)
            
        print(f"\nProcessing Panel ({analysis['panel']}): {analysis['desc']}...")
        x, y, _ = read_two_separate_csvs(str(file_x), str(file_y))
        
        # Determine maximum scale from the 1/4 rule
        windows = generate_fibonacci_windows(len(x) // 4)
        
        # Standard q values for evaluation
        q_values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        
        print("Computing rhoSMFDCCA...")
        results = mfdcca_coefficient(
            x, y, 
            windows=windows, 
            q_values=q_values,
            pol_ord=1
        )
        
        # Format and export
        rows = results_to_rows(results)
        csv_out_path = output_dir / analysis["csv_out"]
        export_to_csv(rows, str(csv_out_path))
        
        csv_paths.append(str(csv_out_path))
        
    # Generate 2-panel Plot
    print("\nGenerating 2-panel vertical publication figure...")
    fig_dir = REPO_ROOT / "figures"
    plot_publication_twopanel_vertical(csv_paths, output_dir=str(fig_dir),
                                       output_name=FIGURE_NAME)

if __name__ == "__main__":
    main()
