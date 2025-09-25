"""
clean_excel.py
- Reads raw Excel files from the raw data directory
- Cleans and processes the data
- Saves results as CSV files in the cleaned directory
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rag_backend folder
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

def clean_excel_file(filepath):
    """
    Process a single Excel file and convert to cleaned CSV
    
    Args:
        filepath: Path to the Excel file
        
    Returns:
        Path to the cleaned CSV file
    """
    filename = os.path.basename(filepath)
    filename_base = os.path.splitext(filename)[0]
    output_csv = os.path.join(CLEAN_DIR, f"{filename_base}_cleaned_long.csv")
    
    print(f"Processing {filename}...")
    
    # Read Excel file
    df = pd.read_excel(filepath)
    
    # Basic cleaning
    # (Adjust this based on your actual data structure)
    df = df.fillna("")
    
    # Example transformation to long format 
    # (Adjust based on your specific Excel structure)
    if "DISTRICT" in df.columns:
        # If already in a good format, just do basic cleaning
        cleaned_df = df.copy()
    else:
        # If needs restructuring, do that here
        # This is just an example - adjust for your actual data
        print("  Restructuring data to long format...")
        # Add your restructuring code here
        cleaned_df = df.copy()  # Placeholder
    
    # Add source information
    cleaned_df["source_file"] = filename
    
    # Save to CSV
    cleaned_df.to_csv(output_csv, index=False)
    print(f"  Saved cleaned data to {output_csv}")
    
    return output_csv

def process_all_excel_files():
    """Process all Excel files in the raw directory"""
    # Ensure the cleaned directory exists
    os.makedirs(CLEAN_DIR, exist_ok=True)
    
    # Find all Excel files
    excel_files = glob.glob(os.path.join(RAW_DIR, "*.xlsx"))
    excel_files += glob.glob(os.path.join(RAW_DIR, "*.xls"))
    
    if not excel_files:
        print(f"No Excel files found in {RAW_DIR}")
        return []
    
    cleaned_files = []
    for excel_file in excel_files:
        cleaned_csv = clean_excel_file(excel_file)
        cleaned_files.append(cleaned_csv)
    
    print(f"\nProcessed {len(excel_files)} files.")
    return cleaned_files

if __name__ == "__main__":
    cleaned_files = process_all_excel_files()
    print("\nAll Excel files processed successfully!")
