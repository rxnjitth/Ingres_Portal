import pandas as pd
import numpy as np

def explore_dataset(file_path):
    """Explore the groundwater dataset to understand its structure"""
    
    print("Loading Excel file...")
    
    # Read all sheets in the Excel file
    try:
        excel_file = pd.ExcelFile(file_path)
        print(f"Excel file contains {len(excel_file.sheet_names)} sheets:")
        for i, sheet in enumerate(excel_file.sheet_names):
            print(f"{i+1}. {sheet}")
        
        # Read the first sheet (or main sheet)
        df = pd.read_excel(file_path, sheet_name=0)
        print(f"\nDataset shape: {df.shape}")
        
        print("\nColumn names:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nBasic statistics:")
        print(df.describe())
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Check for geographic columns
        geo_keywords = ['state', 'district', 'city', 'location', 'area', 'region']
        geo_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in geo_keywords):
                geo_columns.append(col)
        
        if geo_columns:
            print(f"\nPotential geographic columns: {geo_columns}")
            for col in geo_columns:
                print(f"\nUnique values in '{col}':")
                unique_vals = df[col].value_counts()
                print(f"Total unique values: {len(unique_vals)}")
                print(unique_vals.head(10))
        
        # Check for time-related columns
        time_keywords = ['year', 'month', 'date', 'time', 'period']
        time_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in time_keywords):
                time_columns.append(col)
        
        if time_columns:
            print(f"\nPotential time columns: {time_columns}")
            for col in time_columns:
                print(f"\nUnique values in '{col}':")
                unique_vals = df[col].value_counts()
                print(f"Total unique values: {len(unique_vals)}")
                print(unique_vals.head(10))
        
        # Check for water level columns
        water_keywords = ['water', 'level', 'depth', 'ground', 'gw']
        water_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in water_keywords):
                water_columns.append(col)
        
        if water_columns:
            print(f"\nPotential water level columns: {water_columns}")
            for col in water_columns:
                if df[col].dtype in ['float64', 'int64']:
                    print(f"\nStatistics for '{col}':")
                    print(df[col].describe())
        
        return df, excel_file
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, None

if __name__ == "__main__":
    file_path = "CentralReport1758898640827.xlsx"
    df, excel_file = explore_dataset(file_path)