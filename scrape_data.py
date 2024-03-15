import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
from collections import Counter 

df_csv = pd.read_csv('/Users/abdurrehman/Downloads/archive 2/full_data_UpTo_nov2021.csv')
df_csv['scraped_article'] = None
df_titles = df_csv['title']  #! heading
df_description = df_csv['description'] #! summary
df_content = df_csv['content'] #! article


urls = list(df_csv['url'])
urls = urls[:5000]
def scrape_data(url):
    try:
        scraper = cloudscraper.create_scraper()

        response = scraper.get(url)
        response.raise_for_status() # Check for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        class_counts = Counter(tuple(p.get('class', ())) for p in paragraphs)
        most_common_class = max(class_counts, key=class_counts.get)
        main_paragraphs = [p for p in paragraphs if p.get('class', ()) == most_common_class]
        article_content = '\n'.join(p.get_text() for p in main_paragraphs)
        return article_content
    except Exception as e:
        print(f"Error scraping data from {url}: {e}")
        return None

for index, row in df_csv.iterrows():
    article_content = scrape_data(urls[index])
    if article_content:
        df_csv.loc[index, 'scraped_article'] = article_content
        df_csv.to_csv('/Users/abdurrehman/Desktop/NLP/Natural-Language-Processing/new_file.csv', index=False)