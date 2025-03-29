import pandas as pd
import os
"""
File Description:
This script processes translation task datasets with core functionalities:

1. Data Merging:
   - Combines all .parquet files under the Train folder
   - Outputs merged data to train.parquet
   - Automatically resets indexes

2. Data Extraction:
   - Parses 'translation' column from merged data
   - Extracts English (en) and German (de) sentence pairs
   - Automatically filters missing values

3. Data Storage:
   - Saves en/de data as separate .parquet files
   - Storage path: DataProcessing/Train directory
   - Generates file statistics (size, row count)

4. Data Validation:
   - Checks file existence
   - Prints data preview (first 5 rows)
   - Outputs file size, shape, and row statistics

Input Requirements:
- Input files must be in DataProcessing/Train directory
- Files must contain 'translation' column (JSON format)
- Each translation entry must have 'en' and 'de' keys

Outputs:
- Merged file: train.parquet
- English file: en_data_train.parquet
- German file: de_data_train.parquet
- Console statistics and data previews

Notes:
- Ensure consistent encoding for input files
- Empty values are automatically filtered
- Adjust absolute paths according to environment
"""

# Define Train folder containing .parquet files
train_folder = os.path.join('DataProcessing', 'Train')

# Get full paths of all .parquet files in Train folder
train_files = [
    os.path.join(train_folder, file) 
    for file in os.listdir(train_folder) 
    if file.endswith('.parquet')
]

# Merge all parquet files with index reset
combined_train_df = pd.concat(
    [pd.read_parquet(file) for file in train_files], 
    ignore_index=True
)

# Save merged DataFrame to parquet
combined_train_path = os.path.join(train_folder, 'train.parquet')
combined_train_df.to_parquet(combined_train_path, index=False)

# Read merged data for processing
combined_train_df = pd.read_parquet(combined_train_path)

# Extract en/de pairs from translation column
extracted_data = []
for _, row in combined_train_df.iterrows():
    translation = row['translation']
    en_text = translation.get('en', None)
    de_text = translation.get('de', None)
    extracted_data.append({'en': en_text, 'de': de_text})

extracted_df = pd.DataFrame(extracted_data)
print("en/de Columns Preview:")
print(extracted_df.head())

# Define output paths
en_path = os.path.join(train_folder, 'en_data_train.parquet')
de_path = os.path.join(train_folder, 'de_data_train.parquet')

# Save filtered en/de data
extracted_df[['en']].dropna().to_parquet(en_path, index=False)
extracted_df[['de']].dropna().to_parquet(de_path, index=False)

print(f"en data saved to: {en_path}")
print(f"de data saved to: {de_path}")

# File size utility function
def get_file_size(file_path):
    """Get file size in bytes"""
    return os.path.getsize(file_path)

# Validate file statistics
def print_file_stats(file_path, lang):
    """Print file statistics for validation"""
    if not os.path.exists(file_path):
        print(f"{lang} file not found: {file_path}")
        return
    
    file_size = get_file_size(file_path)
    df = pd.read_parquet(file_path)
    row_count, col_count = df.shape
    
    print(f"\n{lang.upper()} File Statistics:")
    print(f"Size: {file_size} bytes")
    print(f"Shape: ({row_count}, {col_count})")
    print(f"Row Count: {row_count}")
    print("First Record:")
    print(df.head(1).to_string(index=False))

# Execute validation
print_file_stats(en_path, 'en')
print_file_stats(de_path, 'de')
