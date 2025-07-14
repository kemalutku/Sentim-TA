from model import *
from datetime import datetime
import os

max_epochs = 1000
batch_size = 32
learning_rate = 1e-3

working_dir = r"D:\CnnTA\v2"

train_dir = os.path.join(working_dir, "data_finance", "train", "1d")
test_dir = os.path.join(working_dir, "data_finance", "test", "1d")

sentiment_ticker = "AAPL"
sentiment_ticker_list = [
    "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "GS", "HD", "IBM",
    "JPM", "MCD", "MMM", "MSFT", "NKE", "TRV", "UNH", "V", "VZ",
]
sentiment_dir = os.path.join(
    working_dir,
    "data_sentim",
    "preprocessed",
    f"{sentiment_ticker}.csv",
)

results_dir = os.path.join(working_dir, "results")
record_dir = os.path.join(working_dir, "records")
checkpoint_dir = os.path.join(working_dir, "checkpoints")

num_parallel_trainings = 1
num_topics = None  # infer from sentiment CSV

model = CnnTa
class_weights = [1, 2, 2]

train_years = [2017, 2022]
test_years = [2023, 2024]

indicators = [
    "RSI", "WIL", "WMA", "EMA", "SMA", "HMA", "TMA", "CCI", "CMO", "MCD",
    "PPO", "ROC", "CMF", "ADX", "PSA",
]

run_name_base = r"DOW30_1h"
run_name = f"{run_name_base}-{datetime.now().strftime('%m_%d_%H_%M')}"
