import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def parse_nodemap(filepath):
    """Parses a single nodemap file to extract metadata and data."""
    metadata = {}
    data_start_line = 0
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    for i, line in enumerate(lines):
        line = line.strip()
        # Check for header first
        if 'ID;' in line:
            data_start_line = i
            break
            
        if line.startswith('#'):
            # Extract metadata
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip('# ').strip()
                val = parts[1].strip()
                metadata[key] = val
    
    # Parse data
    # The header line is like: # ID; x_undef [mm]; ...
    # Wait, in the file view, line 30 is: # ID; x_undef [mm]; ...
    # And line 31 starts data: 0; ...
    
    # Let's find the header line index
    header_line = lines[data_start_line].strip('# ').strip()
    columns = [c.strip() for c in header_line.split(';')]
    
    # output the data block
    data_lines = lines[data_start_line+1:]
    data = []
    for line in data_lines:
        if not line.strip(): continue
        parts = line.strip().split(';')
        if len(parts) == len(columns):
            data.append([float(x) for x in parts])
            
    df = pd.DataFrame(data, columns=columns)
    return metadata, df

def analyze_files(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files.")
    
    # Analyze one file
    sample_file = files[0]
    print(f"Analyzing {sample_file}...")
    meta, df = parse_nodemap(sample_file)
    
    print("Metadata:", meta)
    print("Columns:", df.columns)
    print("Data Range (X):", df['x_undef [mm]'].min(), df['x_undef [mm]'].max())
    print("Data Range (Y):", df['y_undef [mm]'].min(), df['y_undef [mm]'].max())
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df['x_undef [mm]'], df['y_undef [mm]'], c=df['epsy [%]'], cmap='jet', s=1)
    plt.colorbar(label='epsy [%]')
    plt.title(f"Strain field (epsy) - {os.path.basename(sample_file)}")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.axis('equal')
    plt.savefig("analysis_plot.png")
    print("Saved analysis_plot.png")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Choose a directory
    search_path = os.path.join(PROJECT_ROOT, "data", "S_160_2.0", "raw", "Nodemaps")
    analyze_files(search_path)
