import requests
import time
import csv
from datetime import datetime
from tqdm import tqdm
import os

API_KEY = os.environ.get("NYT_API_KEY")
BASE_URL = 'https://api.nytimes.com/svc/archive/v1/{year}/{month}.json'

START_YEAR = 2002
END_YEAR = datetime.now().year
END_MONTH = datetime.now().month

output_file = 'data_sentim/nyt_front_page/articles.csv'

def is_front_page(article):
    # NYT Archive API does not directly indicate front page articles.
    # Heuristics: print_section == 'A' or '1', or section_name == 'A1', or page == '1'
    print_section = article.get('print_section', '')
    section_name = article.get('section_name', '')
    page = article.get('page', '')
    # Check for common front page indicators
    return (
        print_section in ['A', '1'] or
        section_name in ['A1', '1', 'Front Page', 'Page One'] or
        str(page) == '1'
    )

def fetch_articles(year, month):
    url = BASE_URL.format(year=year, month=month)
    response = requests.get(url, params={'api-key': API_KEY})
    if response.status_code == 429:
        print("Rate limit exceeded. Sleeping for 60 seconds.")
        time.sleep(60)
        return fetch_articles(year, month)
    elif response.status_code != 200:
        print(f"Error {response.status_code} for {year}-{month}")
        return []
    data = response.json()
    return data['response']['docs']

def main():
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Headline', 'URL', 'Section'])

        for year in tqdm(range(END_YEAR, START_YEAR - 1, -1), desc="Years"):
            for month in tqdm(range(12, 0, -1), desc=f"Months in {year}", leave=False):
                if year == END_YEAR and month > END_MONTH:
                    continue
                try:
                    articles = fetch_articles(year, month)
                    for article in articles:
                        if is_front_page(article):
                            pub_date = article.get('pub_date', '')
                            headline = article.get('headline', {}).get('main', '')
                            web_url = article.get('web_url', '')
                            section = article.get('section_name', '')
                            writer.writerow([pub_date, headline, web_url, section])
                    time.sleep(6)  # Respect rate limits
                except Exception as e:
                    print(f"Error in {year}-{month}: {e}")
                    continue

if __name__ == '__main__':
    main()
