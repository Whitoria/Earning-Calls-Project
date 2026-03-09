import subprocess
import os
import sys

def run_step(name, script_path):
    print(f"\n--- Step: {name} ---")
    script_abs_path = os.path.abspath(script_path)
    result = subprocess.run([sys.executable, script_abs_path], capture_output=False)
    if result.returncode != 0:
        print(f"Error in {name}. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure we are in the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..") # Move up from SEC/ to root to maintain path consistency
    
    steps = [
        ("Ingestion (Text)", "SEC/src/ingestion/edgar_scraper.py"),
        ("Preprocessing", "SEC/src/preprocessing/text_cleaner.py"),
        ("NLP Metrics", "SEC/src/exploration/nlp_metrics.py"),
        ("Visualizations", "SEC/src/exploration/visualizer.py")
    ]
    
    for name, path in steps:
        run_step(name, path)
        
    print("\n--- Phase 2 Pipeline Complete! ---")
    print("Check SEC/data for Parquet files and SEC/plots for correlation visualizations.")
