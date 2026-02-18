import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Load data
df = pd.read_csv("../RawData/Research_publication_2025.csv")
df.head()
print(df.shape)
# Clean data
df.columns = df.columns.str.lower().str.replace(" ", "_")
df.drop_duplicates(inplace=True)
df['pub_year'] = pd.to_numeric(df['pub_year'], errors='coerce')
df['country'] = df['country'].fillna("Unknown")
df['journal_title'] = df['journal_title'].fillna("Unknown")
df.info()
df.columns.tolist()

#Country Distribution
#country_counts_full = df['country'].value_counts()
country_counts = df['country'].value_counts().head(10)
#country_counts_bottom = df['country'].value_counts().tail(10)
country_counts.plot(kind='bar')
plt.title("Top 10 Publishing Countries")
plt.xlabel("Country")
plt.ylabel("Publications")
for i, v in enumerate(country_counts):
    plt.text(i,v+6000,str(v),ha="center")

plt.savefig("../Reports/Publications/country_distribution.png", bbox_inches='tight')
plt.show()

#Journal Productivity
journal_counts = df['journal_title'].value_counts().head(10)
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10,6))
journal_counts.sort_values().plot(kind='barh', ax=ax)
ax.set_title("Top 10 Journals by Publication Count")
ax.set_xlabel("Number of Publications")
ax.set_ylabel("Journal")
plt.savefig("../Reports/Publications/journal_counts.png", bbox_inches='tight')
plt.show()

#language counts
language_counts=df['lang'].value_counts()
language_counts.plot(kind='bar')
plt.title("Language Distribution")

#plt.savefig("../Reports/Publications/language_distribution.png", bbox_inches='tight')
plt.show()
print(language_counts)

#Topic Mining (NLP Lite)
keywords = ['cancer','virus','diabetes','protein','neuro','alzheimer','infection']
results = []

for k in keywords:
    count = df['pub_title'].str.contains(k, case=False, na=False).sum()
    results.append((k, count))

topic_df = pd.DataFrame(results, columns=['Keyword','Count'])
topic_df

#Topic Visualization
topic_df.plot(kind='bar', x='Keyword', y='Count')
plt.title("Research Topics Distribution")
plt.savefig("../Reports/Publications/topic_analysis.png", bbox_inches='tight')
plt.show()

#Author Collaboration Analysis
#df['author_count'] = df['author_list'].str.split(';').apply(len)
df['author_count'] = (
    df['author_list']
    .fillna('')
    .str.count(';') + 1
)

# Fix empty rows
df.loc[df['author_list'].isna(), 'author_count'] = 0
print("Average Authors per Paper:", df['author_count'].mean())
print(df['author_count'])
bins = list(range(0, 101, 10))
plt.figure(figsize=(8,5))
plt.hist(
    df['author_count'],
    bins=bins,
    rwidth=0.85
)
plt.title("Author Collaboration Distribution")
plt.xlabel("Number of Authors")
plt.ylabel("Number of Publications")
plt.savefig("../Reports/Publications/collaboration.png", bbox_inches='tight')
plt.show()

#memory optimization
df.info(memory_usage='deep')
df['country'] = df['country'].astype('category')
df['journal_title'] = df['journal_title'].astype('category')
df.info(memory_usage='deep')