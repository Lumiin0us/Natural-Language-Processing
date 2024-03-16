import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

# pattern = '''Log in to comment on videos and join in on the fun.
# Watch the live stream of Fox News and full episodes.
# Reduce eye strain and focus on the content that matters.
# ©2024 FOX News Network, LLC. All rights reserved. This material may not be published, broadcast, rewritten, or redistributed. All market data delayed 20 minutes.'''.strip()

df = pd.read_csv('scraped_datafile.csv')
df.drop(['content', 'source', 'author', 'Unnamed: 0', 'url', 'publishedAt', 'requested_date'], axis=1, inplace=True)
df.dropna(inplace=True)
df['scraped_article'] = df['scraped_article'].replace('\t', '').replace('\\n', '', regex=True)
df['scraped_article'] = df['scraped_article'].str.strip()
word_count = df['scraped_article'].apply(lambda x: len(x.split()))

rows_to_drop = (word_count <= 100)
df = df[~rows_to_drop]
# print(df['scraped_article'].iloc[115])
# print(len(word_count))
# print(len(df['scraped_article'].iloc[115]))
# new_df = df['scraped_article'].iloc[172]
# new_df = new_df.strip()
# print(len(new_df))
# df.to_csv('/Users/abdurrehman/Desktop/NLP/Natural-Language-Processing/dataset.csv', index=False)