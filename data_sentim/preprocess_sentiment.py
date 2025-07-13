import os
import pandas as pd
import glob
import numpy as np
from collections import defaultdict
import re
from datetime import datetime
import time
import argparse

# Directory paths (update as needed)
finance_data_dir = r"D:/CnnTA/v2/data_finance"
finance_train_dir = os.path.join(finance_data_dir, "train", "1d")
finance_test_dir = os.path.join(finance_data_dir, "test", "1d")
sentiment_data_dir = r"D:\CnnTA\v2\data_sentim\raw"
output_dir = r"D:/CnnTA/v2/data_sentim"
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Preprocess sentiment data")
parser.add_argument(
    "--count-only",
    action="store_true",
    help="aggregate topic counts instead of sentiment scores",
)
args = parser.parse_args()
COUNT_ONLY = args.count_only

# Mapping from finance symbol to sentiment company name
sentiment_to_finance = {
    '3m': 'MMM',
    'amazon': 'AMZN',
    'american_express': 'AXP', 
    'amgen': 'AMGN',
    'apple': 'AAPL',
    'boeing': 'BA',
    'caterpillar': 'CAT',
    'chevron': 'CVX',
    'cisco_systems': 'CSCO',
    'disney': 'DIS',
    'goldman_sachs': 'GS',
    'home_depot': 'HD',
    'honeywell_international': 'HON',
    'ibm': 'IBM',
    'jpmorgan_chase': 'JPM',
    "mcdonald's": 'MCD',
    'microsoft': 'MSFT',
    'nike': 'NKE',
    'nvidia': 'NVDA',
    'salesforce': 'CRM',
    'travelers': 'TRV',
    'unitedhealth_group': 'UNH',
    'verizon_communications': 'VZ',
    'visa': 'V',
}

sentim_csvs = glob.glob(os.path.join(sentiment_data_dir, "*.csv"))

for sc in sentim_csvs:
    # Extract company name from filename (between gdelt_ and .csv)
    filename = os.path.basename(sc)
    company_name = filename.replace("gdelt_", "").replace(".csv", "")
    
    # Find corresponding finance ticker
    finance_ticker = sentiment_to_finance.get(company_name)
    
    if finance_ticker:
        # Find corresponding finance data files
        finance_train_pattern = os.path.join(finance_train_dir, f"{finance_ticker}_*.csv")
        finance_test_pattern = os.path.join(finance_test_dir, f"{finance_ticker}_*.csv")

        try:
            finance_train_file = glob.glob(finance_train_pattern)[0]
            finance_test_file = glob.glob(finance_test_pattern)[0]
        except IndexError:
            print(f"Finance data files not found for {company_name} ({finance_ticker}). Skipping...")
            continue 
        
        print(f"Processing {company_name} -> {finance_ticker}")
    else:
        raise ValueError("Sentiment company name did not match with finance ticker.")
    
    sentiment_df = pd.read_csv(sc).rename(columns={'Headline': 'headline'})
    sentiment_df['headline'] = sentiment_df['headline'].str.lower()

    sentiment_folder = os.path.basename(sc).replace(".csv", "").replace("gdelt", "sentiment_gdelt")
    sentiment_folder = os.path.join(sentiment_data_dir, sentiment_folder)

    headline_csv_path = os.path.join(sentiment_folder, "headline_topics.csv")
    sentiment_csv_path = os.path.join(sentiment_folder, "headline_sentiments.csv")

    headline_df = pd.read_csv(headline_csv_path)
    sentiment_score_df = pd.read_csv(sentiment_csv_path)

    sentiment_df = pd.merge(sentiment_df, headline_df, on='headline', how='inner')
    sentiment_df = pd.merge(sentiment_df, sentiment_score_df, on='headline', how='inner')
    sentiment_df = sentiment_df.drop(columns=['URL', 'MobileURL', 'headline'])

    merged_df_path = os.path.join(output_dir, "merged", os.path.basename(sc) + ".csv")
    sentiment_df.to_csv(merged_df_path, index=False)

    merged_df = sentiment_df.copy()
    
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    # Convert to daily timestamp (days since epoch) - simpler approach
    merged_df['date_timestamp'] = (merged_df['Date'] - pd.Timestamp('1970-01-01')).dt.days

    topic_columns = [f't{i}' for i in range(15)]
    
    # Vectorized aggregation using pivot_table - much faster than nested loops
    if COUNT_ONLY:
        aggregated_df = merged_df.pivot_table(
            index='date_timestamp',
            columns='topic_id',
            values='sentiment',
            aggfunc='count',
            fill_value=0
        ).reset_index()
    else:
        aggregated_df = merged_df.pivot_table(
            index='date_timestamp',
            columns='topic_id',
            values='sentiment',
            aggfunc='sum',
            fill_value=0
        ).reset_index()

    # Rename columns to match expected format
    aggregated_df.columns = ['date'] + [f't{int(col)}' for col in aggregated_df.columns[1:]]

    # Ensure all topics 0-14 are present, add missing ones with 0 values
    for topic_id in range(15):
        if f't{topic_id}' not in aggregated_df.columns:
            aggregated_df[f't{topic_id}'] = 0

    # Reorder columns to ensure consistent order
    column_order = ['date'] + [f't{i}' for i in range(15)]
    aggregated_df = aggregated_df[column_order]

    # Convert date back to int
    aggregated_df['date'] = aggregated_df['date'].astype(int)

    # Sort by date
    aggregated_df = aggregated_df.sort_values('date').reset_index(drop=True)
    
    suffix = "_count" if COUNT_ONLY else ""
    aggregated_df_path = os.path.join(
        output_dir, "aggregated", os.path.basename(sc).replace(".csv", f"{suffix}.csv")
    )
    os.makedirs(os.path.dirname(aggregated_df_path), exist_ok=True)
    aggregated_df.to_csv(aggregated_df_path, index=False)

    finance_train_df = pd.read_csv(finance_train_file).rename(columns={'Date': 'date'})
    finance_test_df = pd.read_csv(finance_test_file).rename(columns={'Date': 'date'})

    # Combine finance train and test data
    finance_df = pd.concat([finance_train_df, finance_test_df], ignore_index=True)

    # Convert finance dates from milliseconds to daily timestamps
    finance_df['date'] = pd.to_datetime(finance_df['date'], unit='ms')
    finance_df['date'] = (finance_df['date'] - pd.Timestamp('1970-01-01')).dt.days

    # Forward-fill sentiment data to next available finance day
    # This handles weekends and holidays where there's sentiment but no finance data
    finance_dates_set = set(finance_df['date'].unique())

    # Create a copy of aggregated_df to modify
    adjusted_sentiment_df = aggregated_df.copy()

    # For each sentiment date not in finance data, find the next available finance date
    sentiment_dates_to_adjust = []
    for sent_date in sorted(aggregated_df['date'].unique()):
        if sent_date not in finance_dates_set:
            # Find the next available finance date
            next_finance_date = None
            for fin_date in sorted(finance_dates_set):
                if fin_date > sent_date:
                    next_finance_date = fin_date
                    break

            if next_finance_date is not None:
                sentiment_dates_to_adjust.append((sent_date, next_finance_date))

    # Aggregate sentiment data from missing days to next available finance day
    topic_columns = [f't{i}' for i in range(15)]

    for original_date, target_date in sentiment_dates_to_adjust:
        # Get the sentiment data for the original date
        original_row = aggregated_df[aggregated_df['date'] == original_date].iloc[0]

        # Check if target date already exists in adjusted_sentiment_df
        if target_date in adjusted_sentiment_df['date'].values:
            # Add sentiment values to existing target date
            target_idx = adjusted_sentiment_df[adjusted_sentiment_df['date'] == target_date].index[0]
            for topic_col in topic_columns:
                adjusted_sentiment_df.loc[target_idx, topic_col] += original_row[topic_col]
        else:
            # Create new row for target date with original sentiment values
            new_row = original_row.copy()
            new_row['date'] = target_date
            adjusted_sentiment_df = pd.concat([adjusted_sentiment_df, pd.DataFrame([new_row])], ignore_index=True)

        # Remove the original date row since it's been moved
        adjusted_sentiment_df = adjusted_sentiment_df[adjusted_sentiment_df['date'] != original_date]

    # Sort by date and reset index
    adjusted_sentiment_df = adjusted_sentiment_df.sort_values('date').reset_index(drop=True)

    # Update aggregated_df with the adjusted version
    aggregated_df = adjusted_sentiment_df

    # Convert date back to milliseconds since epoch to match finance data
    aggregated_df['date'] = aggregated_df['date'].astype(int) * 86400000
    
    # Save the adjusted aggregated data with the stock ticker as filename
    ticker_suffix = "_count" if COUNT_ONLY else ""
    ticker_filename = f"{finance_ticker}{ticker_suffix}.csv"
    aggregated_df_path = os.path.join(output_dir, "preprocessed", ticker_filename)
    os.makedirs(os.path.dirname(aggregated_df_path), exist_ok=True)
    aggregated_df.to_csv(aggregated_df_path, index=False)

    print(f"Adjusted {len(sentiment_dates_to_adjust)} sentiment dates to next available finance days")
    if sentiment_dates_to_adjust:
        print(f"  Sample adjustments: {sentiment_dates_to_adjust[:3]}")

    # Print structure of finance data
    print(f"\nFinance data structure for {finance_ticker}:")
    print(f"  Columns: {list(finance_df.columns)}")
    print(f"  Shape: {finance_df.shape}")
    print(f"  Date range: {finance_df['date'].min()} to {finance_df['date'].max()}")

    # Print structure of aggregated sentiment data
    print(f"\nAggregated sentiment data structure for {company_name}:")
    print(f"  Columns: {list(aggregated_df.columns)}")
    print(f"  Shape: {aggregated_df.shape}")
    print(f"  Date range: {aggregated_df['date'].min()} to {aggregated_df['date'].max()}")

    # Get unique dates from both datasets
    finance_dates = set(finance_df['date'].unique())
    sentiment_dates = set(aggregated_df['date'].unique())

    # Find differences
    finance_only = finance_dates - sentiment_dates
    sentiment_only = sentiment_dates - finance_dates
    common_dates = finance_dates & sentiment_dates

    print(f"\nDate matching analysis for {company_name} ({finance_ticker}):")
    print(f"  Total finance dates: {len(finance_dates)}")
    print(f"  Total sentiment dates: {len(sentiment_dates)}")
    print(f"  Common dates: {len(common_dates)}")
    print(f"  Finance dates not in sentiment: {len(finance_only)}")
    print(f"  Sentiment dates not in finance: {len(sentiment_only)}")

    if len(finance_only) > 0:
        print(f"  Sample finance-only dates: {sorted(list(finance_only))[:5]}")
    if len(sentiment_only) > 0:
        print(f"  Sample sentiment-only dates: {sorted(list(sentiment_only))[:5]}")
    print("-" * 80)
