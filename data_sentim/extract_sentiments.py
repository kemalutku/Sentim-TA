import os, time, json, pickle, random, re
from datetime import datetime
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# --------  global hyper-params  -------------------------------------
EMBEDDING_BATCH_SIZE = 512
SENTIMENT_BATCH_SIZE = 256
N_CLUST_SCAN = 50_000
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# --------  load models ONCE; reuse for every file  ------------------
log("Loading models (once)")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

tok = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD_ID = tok.pad_token_id
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model = (AutoModelForCausalLM
         .from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                          torch_dtype=torch.float16)
         .to("cuda")
         .eval())
model.config.use_cache = False


#######################################################################
# 1.  Helper functions  (unchanged from your script)
#######################################################################
def clean_one(raw: str) -> str:
    first = raw.strip().splitlines()[0]
    first = re.sub(r"^\d+[\).]\s*", "", first)
    first = re.sub(r"[^\w\s&\-:]", "", first)
    return " ".join(first.split()[:6]).title()


def batched_micro_titles(texts, bs=256):
    out = []
    for i in tqdm(range(0, len(texts), bs), desc="micro-titles", leave=False):
        prompts = [("Give a concise 7-15 word topic for the headline, if your summary topic is about a company, "
                    "describe the related event / action / item that is happening instead of just saying "
                    "for example: Apple Stocks. "
                    "no bullet points or extra words.\n\n"
                    f'"{h}"\nTopic:') for h in texts[i:i + bs]]
        toks = tok(prompts, return_tensors="pt",
                   padding=True, truncation=True, max_length=96).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**toks, max_new_tokens=8,
                                 pad_token_id=PAD_ID, do_sample=False)
        outs = tok.batch_decode(gen[:, toks["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        out.extend(map(clean_one, outs))
    return out


def batched_sentiment(texts, bs=128):
    out, total, ptr = [], len(texts), 0
    pbar = tqdm(total=total, desc="sentiment", leave=False)

    while ptr < total:
        cur_bs = min(bs, total - ptr)
        while True:
            try:
                batch = texts[ptr:ptr + cur_bs]
                prompts = [(
                    "Score the sentiment between -1.0 and 1.0. "
                    "-1 = very negative, 1 = very positive. "
                    "Return only the number.\n\n"
                    f'"{h}"\nScore:'
                ) for h in batch]

                toks = tok(prompts, return_tensors="pt",
                           padding=True, truncation=True,
                           max_length=96).to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**toks, max_new_tokens=4,
                                         pad_token_id=PAD_ID, do_sample=False)
                break  # success
            except torch.cuda.OutOfMemoryError:
                cur_bs //= 2
                torch.cuda.empty_cache()
                if cur_bs < 8:
                    raise

        replies = tok.batch_decode(
            gen[:, toks["input_ids"].shape[1]:],
            skip_special_tokens=True)

        for r in replies:
            try:
                out.append(float(r.strip().split()[0]))
            except ValueError:
                out.append(0.0)

        ptr += cur_bs
        pbar.update(cur_bs)

        del toks, gen
        torch.cuda.empty_cache()

    pbar.close()
    return np.clip(out, -1.0, 1.0).tolist()


#######################################################################
# 2.  The full pipeline for ONE CSV
#######################################################################
def run_pipeline(RAW_CSV: str, OUT_DIR: str, start_date: str = None, end_date: str = None):
    """
    Process one CSV file with optional date range filtering.
    
    Args:
        RAW_CSV: Path to input CSV file
        OUT_DIR: Output directory for results
        start_date: Start date filter (format: 'YYYY-MM-DD', inclusive)
        end_date: End date filter (format: 'YYYY-MM-DD', inclusive)
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"=== Processing {os.path.basename(RAW_CSV)} ===")

    # ---------- ingest ------------------------------------------------
    df = pd.read_csv(RAW_CSV)
    
    # Check if required columns exist
    required_columns = ['Date', 'Headline']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        log(f"Error: Missing required columns {missing_columns} in {os.path.basename(RAW_CSV)}")
        log(f"Available columns: {list(df.columns)}")
        return
    
    log(f"Loaded {len(df)} rows from {os.path.basename(RAW_CSV)}")
    
    # Apply date filtering if specified
    if start_date or end_date:
        log(f"Applying date filter: {start_date or 'no start'} to {end_date or 'no end'}")
        
        # Convert Date column to datetime, handling different formats
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Remove rows with invalid dates
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                log(f"Warning: {invalid_dates} rows have invalid dates and will be excluded")
                df = df.dropna(subset=['Date'])
        except Exception as e:
            log(f"Error parsing dates: {e}")
            log("Skipping date filtering for this file")
            return
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            # Handle timezone compatibility
            if df['Date'].dt.tz is not None:
                start_dt = start_dt.tz_localize('UTC')
            df = df[df['Date'] >= start_dt]
            log(f"After start date filter: {len(df)} rows")
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            # Handle timezone compatibility
            if df['Date'].dt.tz is not None:
                end_dt = end_dt.tz_localize('UTC')
            df = df[df['Date'] <= end_dt]
            log(f"After end date filter: {len(df)} rows")
    
    if len(df) == 0:
        log("No data left after filtering. Skipping...")
        return
    
    df["headline"] = df["Headline"].astype(str).str.strip().str.lower()
    heads = df["headline"].tolist()

    # ---------- embeddings -------------------------------------------
    emb_file = f"{OUT_DIR}/embeddings.npy"
    if os.path.exists(emb_file):
        emb = np.load(emb_file)
    else:
        log("Embedding headlines")
        emb = embedder.encode(heads, batch_size=EMBEDDING_BATCH_SIZE,
                              convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=True)
        np.save(emb_file, emb)

    # ---------- clustering -------------------------------------------
    kmeans_pkl = f"{OUT_DIR}/kmeans.pkl"
    topic_csv = f"{OUT_DIR}/headline_topics.csv"
    if os.path.exists(topic_csv):
        df["topic_id"] = pd.read_csv(topic_csv)["topic_id"]
    else:
        if os.path.exists(kmeans_pkl):
            mbk = pickle.load(open(kmeans_pkl, "rb"))
            df["topic_id"] = mbk.predict(emb)
        else:
            log("Picking K")
            sample = emb[np.random.choice(len(emb), min(N_CLUST_SCAN, len(emb)), False)]
            # best_k, best_sc = None, -1
            # for k in tqdm(range(6, 21), desc="K-scan", leave=False):
            #     mbk = MiniBatchKMeans(k, batch_size=8192, n_init=10, max_iter=100, random_state=42).fit(sample)
            #     sc = silhouette_score(sample, mbk.labels_)
            #     if sc > best_sc: best_k, best_sc = k, sc
            # log(f"→ K={best_k}")
            best_k = 15 # Pick k=15 for match 15x15 images of CNN_TA implementation
            mbk = MiniBatchKMeans(best_k, batch_size=8192, n_init=10, max_iter=300, random_state=42)
            df["topic_id"] = mbk.fit_predict(emb)
            pickle.dump(mbk, open(kmeans_pkl, "wb"))
        df[["headline", "topic_id"]].to_csv(topic_csv, index=False)

    # ---------- micro-titles & macro label ----------------------------
    micro_csv = f"{OUT_DIR}/headline_micro_titles.csv"
    if os.path.exists(micro_csv):
        df["micro_title"] = pd.read_csv(micro_csv)["micro_title"]
    else:
        log("Generating micro-titles")
        df["micro_title"] = batched_micro_titles(df["headline"])
        df[["headline", "micro_title"]].to_csv(micro_csv, index=False)

    topic_names = (df.groupby("topic_id")["micro_title"]
                   .apply(lambda s: Counter(s).most_common(1)[0][0])
                   .to_dict())
    json.dump({int(k): v for k, v in topic_names.items()},
              open(f"{OUT_DIR}/topic_names.json", "w"), ensure_ascii=False, indent=2)
    df["topic"] = df.topic_id.map(topic_names)

    # ---------- sentiment --------------------------------------------
    sent_csv = f"{OUT_DIR}/headline_sentiments.csv"
    if os.path.exists(sent_csv):
        df["sentiment"] = pd.read_csv(sent_csv)["sentiment"]
    else:
        log("Running sentiment")
        df["sentiment"] = batched_sentiment(heads)
        df[["headline", "sentiment"]].to_csv(sent_csv, index=False)

    # ---------- top-10 table -----------------------------------------
    top10_csv = f"{OUT_DIR}/top10_sentiment_table.csv"
    if not os.path.exists(top10_csv):
        top_ids = (df.groupby("topic").size()
                   .sort_values(ascending=False).head(10).index)
        top10 = (df[df["topic"].isin(top_ids)]
                 .pivot_table(index="topic", columns="sentiment",
                              values="headline", aggfunc="count",
                              fill_value=0)
                 .assign(total=lambda x: x.sum(1))
                 .sort_values("total", ascending=False))
        top10.to_csv(top10_csv)

    log("Done → results in " + OUT_DIR)


#######################################################################
# 3.  Driver loop over every CSV in the folder
#######################################################################
if __name__ == "__main__":
    # Date range configuration
    START_DATE = "2017-01-01"  # Set to None to disable start date filter
    END_DATE = "2024-12-31"    # Set to None to disable end date filter

    DATA_DIR = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Okul\NLP\SentimTA\data_sentim\cleaned"

    log(f"Processing files with date range: {START_DATE} to {END_DATE}")
    
    # Process only nyt_articles.csv
    csv_path = "nyt_articles.csv"
    if os.path.exists(os.path.join(DATA_DIR, csv_path)):
        raw_csv = os.path.join(DATA_DIR, csv_path)
        
        # Add date range suffix to output directory name
        date_suffix = ""
        if START_DATE or END_DATE:
            start_str = START_DATE.replace("-", "") if START_DATE else "start"
            end_str = END_DATE.replace("-", "") if END_DATE else "end"
            date_suffix = f"_{start_str}_to_{end_str}"
        
        out_dir = os.path.join(DATA_DIR, f"sentiment_{os.path.splitext(csv_path)[0]}{date_suffix}")
        run_pipeline(raw_csv, out_dir, START_DATE, END_DATE)
    else:
        log(f"File {csv_path} not found in {DATA_DIR}")
    
    # Uncomment below to process all CSV files
    # for csv_path in sorted(os.listdir(DATA_DIR)):
    #     if not csv_path.lower().endswith(".csv"):
    #         continue
    #     raw_csv = os.path.join(DATA_DIR, csv_path)
    #     
    #     # Add date range suffix to output directory name
    #     date_suffix = ""
    #     if START_DATE or END_DATE:
    #         start_str = START_DATE.replace("-", "") if START_DATE else "start"
    #         end_str = END_DATE.replace("-", "") if END_DATE else "end"
    #         date_suffix = f"_{start_str}_to_{end_str}"
    #     
    #     out_dir = os.path.join(DATA_DIR, f"sentiment_{os.path.splitext(csv_path)[0]}{date_suffix}")
    #     run_pipeline(raw_csv, out_dir, START_DATE, END_DATE)
