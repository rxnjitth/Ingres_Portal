import pandas as pd
import numpy as np

def parse_groundwater_data(file_path):
    """Parse the groundwater Excel file with proper header detection"""
    
    # Read the raw Excel data first
    df_raw = pd.read_excel(file_path, sheet_name=0, header=None)
    
    print("Raw data shape:", df_raw.shape)
    print("\nFirst 20 rows of raw data:")
    print(df_raw.iloc[:20, :10])  # Show first 20 rows, 10 columns
    
    # Find where the actual data starts by looking for headers
    header_row = None
    data_start_row = None
    
    for i in range(min(20, len(df_raw))):
        row_values = df_raw.iloc[i].astype(str).str.lower()
        if any(['district' in str(val) for val in row_values]):
            print(f"Found potential header at row {i}")
            print("Row content:", df_raw.iloc[i].values[:10])
            header_row = i
            data_start_row = i + 1
            break
    
    if header_row is None:
        # Try to find data by looking for state names
        indian_states = ['andhra pradesh', 'assam', 'bihar', 'gujarat', 'haryana', 'karnataka', 
                        'kerala', 'madhya pradesh', 'maharashtra', 'punjab', 'rajasthan', 
                        'tamil nadu', 'uttar pradesh', 'west bengal']
        
        for i in range(min(50, len(df_raw))):
            row_str = ' '.join(df_raw.iloc[i].astype(str)).lower()
            if any(state in row_str for state in indian_states):
                print(f"Found data starting around row {i}")
                print("Row content:", df_raw.iloc[i].values[:10])
                if i > 0:
                    header_row = i - 1
                    data_start_row = i
                break
    
    # Try different approaches to read the data
    if header_row is not None:
        print(f"\nTrying to read data with header at row {header_row}")
        df = pd.read_excel(file_path, sheet_name=0, header=header_row, skiprows=range(0, header_row))
    else:
        # If we can't find proper headers, try skipping some rows
        print("\nTrying to read data by skipping first few rows")
        for skip_rows in [5, 10, 15, 20]:
            try:
                df = pd.read_excel(file_path, sheet_name=0, skiprows=skip_rows)
                if len(df) > 0 and not df.columns[0].startswith('Unnamed'):
                    print(f"Successfully read data by skipping {skip_rows} rows")
                    break
            except:
                continue
        else:
            # Last resort - read all data and clean manually
            df = pd.read_excel(file_path, sheet_name=0)
    
    print(f"\nProcessed data shape: {df.shape}")
    print("\nColumn names:")
    for i, col in enumerate(df.columns[:20]):  # Show first 20 columns
        print(f"{i+1}. {col}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    # Look for geographic and water level data
    print("\nAnalyzing data patterns...")
    
    # Check each column for data type and content
    for i, col in enumerate(df.columns[:20]):  # Analyze first 20 columns
        non_null_count = df[col].count()
        if non_null_count > 0:
            sample_values = df[col].dropna().head(5).values
            print(f"Column {i+1} '{col}': {non_null_count} non-null values, sample: {sample_values}")
    
    return df

if __name__ == "__main__":
    file_path = "CentralReport1758898640827.xlsx"
    df = parse_groundwater_data(file_path)