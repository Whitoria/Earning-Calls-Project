import pandas as pd
import os

DATA_DIR = "SEC/data"

def generate_fetched_summary():
    OUTPUT_CSV = "SEC/fetched_data_structure_summary.csv"
    fetched_files = [
        f"{DATA_DIR}/raw_filings_metadata.parquet",
        f"{DATA_DIR}/section_text.parquet",
        f"{DATA_DIR}/defeatbeta_financials.parquet"
    ]
    
    summary_data = []
    for file_path in fetched_files:
        if not os.path.exists(file_path):
            continue
        filename = os.path.basename(file_path)
        try:
            df = pd.read_parquet(file_path)
            rows, cols = df.shape
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null_count = df[col].notna().sum()
                sample_val = ""
                if col in ["section_name", "filing_type", "metric", "ticker", "form"]:
                    unique_vals = list(df[col].dropna().unique())
                    if len(unique_vals) > 0:
                        sample_val = " | ".join([str(v) for v in unique_vals])
                elif non_null_count > 0:
                    first_valid_idx = df[col].first_valid_index()
                    if first_valid_idx is not None:
                        val = df.loc[first_valid_idx, col]
                        if isinstance(val, str) and len(val) > 500:
                            sample_val = val[:500] + "... [Text truncated]"
                        else:
                            sample_val = str(val)
                summary_data.append({
                    "File Name": filename, "Total Rows in File": rows, "Total Cols in File": cols,
                    "Column Name": col, "Data Type": dtype, "Non-Null Count": non_null_count,
                    "Sample Categories / Value": sample_val
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    pd.DataFrame(summary_data).to_csv(OUTPUT_CSV, index=False)
    print(f"Fetched summary generated at: {OUTPUT_CSV}")

def generate_engineered_summary():
    OUTPUT_CSV = "SEC/engineered_data_structure_summary.csv"
    engineered_files = [
        f"{DATA_DIR}/cleaned_section_text.parquet",
        f"{DATA_DIR}/nlp_metrics.parquet",
        f"{DATA_DIR}/keyword_year_table.parquet",
        f"{DATA_DIR}/financial_growth_metrics.parquet",
        f"{DATA_DIR}/correlated_data.parquet"
    ]
    
    summary_data = []
    for file_path in engineered_files:
        if not os.path.exists(file_path):
            continue
        filename = os.path.basename(file_path)
        try:
            df = pd.read_parquet(file_path)
            rows, cols = df.shape
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null_count = df[col].notna().sum()
                sample_val = ""
                if col in ["section_name", "filing_type", "metric", "ticker"]:
                    unique_vals = list(df[col].dropna().unique())
                    if len(unique_vals) > 0:
                        sample_val = " | ".join([str(v) for v in unique_vals])
                elif non_null_count > 0:
                    first_valid_idx = df[col].first_valid_index()
                    if first_valid_idx is not None:
                        val = df.loc[first_valid_idx, col]
                        if isinstance(val, str) and len(val) > 500:
                            sample_val = val[:500] + "... [Text truncated]"
                        else:
                            sample_val = str(val)
                summary_data.append({
                    "File Name": filename, "Total Rows in File": rows, "Total Cols in File": cols,
                    "Column Name": col, "Data Type": dtype, "Non-Null Count": non_null_count,
                    "Sample Categories / Value": sample_val
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    pd.DataFrame(summary_data).to_csv(OUTPUT_CSV, index=False)
    print(f"Engineered summary generated at: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_fetched_summary()
    generate_engineered_summary()
