import os
import pandas as pd

# Directories
GDELT_DIR = os.path.join('data_sentim', 'gdelt')
NYT_DIR = os.path.join('data_sentim', 'nyt_front_page')
CLEANED_DIR = os.path.join('data_sentim', 'cleaned')

os.makedirs(CLEANED_DIR, exist_ok=True)

def clean_duplicates_in_folder(folder_path, cleaned_dir):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            # Determine which column to use for title deduplication
            if folder_path.endswith('gdelt'):
                title_col = 'Title'
            elif folder_path.endswith('nyt_front_page'):
                title_col = 'Headline'
            else:
                print(f"Unknown folder structure for {folder_path}. Skipping {filename}.")
                continue
            if title_col in df.columns:
                original_count = len(df)
                df_cleaned = df.drop_duplicates(subset=[title_col])
                cleaned_count = len(df_cleaned)
                removed_count = original_count - cleaned_count
            else:
                print(f"Warning: '{title_col}' column not found in {filename}. Skipping.")
                continue
            cleaned_path = os.path.join(cleaned_dir, filename)
            df_cleaned.to_csv(cleaned_path, index=False)
            print(f"Cleaned file saved to {cleaned_path}")
            print(f"{filename}: {removed_count} duplicates removed, {cleaned_count} remaining (from {original_count})")

def main():
    clean_duplicates_in_folder(GDELT_DIR, CLEANED_DIR)
    clean_duplicates_in_folder(NYT_DIR, CLEANED_DIR)

if __name__ == "__main__":
    main()
