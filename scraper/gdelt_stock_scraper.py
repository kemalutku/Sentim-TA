#!/usr/bin/env python3
import requests, csv, io, time
from datetime import datetime, timedelta
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Dow 30 components (as of Jan 11 2025) – source: German Wikipedia & FT.com
DJ30_COMPANIES = [
    "Goldman Sachs", "UnitedHealth Group", "Microsoft", "Home Depot",
    "Caterpillar", "Sherwin-Williams", "Salesforce", "Visa",
    "American Express", "McDonald's", "Amgen", "JPMorgan Chase",
    "Apple", "Travelers", "IBM", "Amazon", "Honeywell International",
    "Boeing", "Procter & Gamble", "Chevron", "Johnson & Johnson",
    "Nvidia", "3M", "Disney", "Merck & Co.", "Walmart", "Nike",
    "Coca-Cola", "Cisco Systems", "Verizon Communications"
]

# GDELT DOC 2.0 endpoint & params
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS   = 250        # ArtList cap
RETRY_BACKOFF = 5          # sec on HTTP 429
START_DATETIME = "20170101000000"
END_DATETIME   = datetime.now().strftime("%Y%m%d") + "235959"
# ─────────────────────────────────────────────────────────────

def fetch_articles(query, start_dt, end_dt):
    """Fetch up to MAX_RECORDS articles between start_dt and end_dt, retry on 429."""
    params = {
        "query":         query,
        "mode":          "ArtList",
        "format":        "CSV",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end_dt.strftime("%Y%m%d%H%M%S"),
        "maxrecords":    MAX_RECORDS
    }
    backoff = RETRY_BACKOFF
    while True:
        resp = requests.get(GDELT_API_URL, params=params,
                            headers={"User-Agent": "GDELTNewsFetcher/1.0"})
        if resp.status_code == 429:
            time.sleep(backoff); backoff *= 2
            continue
        resp.raise_for_status()
        text = resp.content.decode("utf-8-sig")  # strip BOM
        buf  = io.StringIO(text)
        reader = csv.reader(buf)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        return header, list(reader)

def month_windows(start, end):
    """Yield (window_start, window_end) for every calendar month in [start,end]."""
    cur = start
    while cur <= end:
        if cur.month == 12:
            nxt = datetime(cur.year+1, 1, 1)
        else:
            nxt = datetime(cur.year, cur.month+1, 1)
        win_end = min(end, nxt - timedelta(seconds=1))
        yield cur, win_end
        cur = nxt

def run_for_company(company):
    query = f"{company} sourcelang:english"
    out_file = f"gdelt_{company.lower().replace(' ', '_').replace('&','and')}.csv"
    start    = datetime.strptime(START_DATETIME, "%Y%m%d%H%M%S")
    end      = datetime.strptime(END_DATETIME,   "%Y%m%d%H%M%S")

    seen     = set()
    writer   = None
    url_i    = date_i = None

    # iterate months with inner progress bar

    for ws, we in tqdm(list(month_windows(start, end)),
                       desc=f"{company[:12]:12}", unit="month", leave=False):
        header, rows = fetch_articles(query, ws, we)
        if not header:
            continue

        # initialize writer & find cols on first batch
        if writer is None:
            header[0] = header[0].lstrip("\ufeff")
            lc = [h.lower() for h in header]
            # Find URL column robustly
            url_i = next((i for i, h in enumerate(lc) if "url" in h), None)
            if url_i is None:
                tqdm.write(f"ERROR: No URL column found in header: {header}. Skipping batch.")
                continue
            # Find date column robustly
            if "seendate" in lc:
                date_i = lc.index("seendate")
            else:
                date_i = next((i for i, h in enumerate(lc) if "date" in h), None)
            if date_i is None:
                tqdm.write(f"ERROR: No date column found in header: {header}. Skipping batch.")
                continue

            f = open(out_file, "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(header)

        new_count = 0
        for row in rows:
            key = (row[url_i], row[date_i])
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(row)
            new_count += 1

        # annotate the inner bar
        tqdm.write(f"{company:12} {ws.strftime('%Y-%m')} → "
                   f"fetched {len(rows):3} rows, +{new_count:3} new "
                   f"(total {len(seen):4})")

    print(f"{company:12} → saved {len(seen)} unique articles to {out_file}")

def main():
    for company in tqdm(DJ30_COMPANIES, desc="Companies", unit="company"):
        run_for_company(company)

if __name__ == "__main__":
    main()
