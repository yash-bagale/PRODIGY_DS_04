# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter

nltk.download('stopwords')

# -----------------------------
# Load Dataset (Fix Headers)
# -----------------------------
df = pd.read_csv("twitter_training.csv", header=None)

# Rename columns properly
df.columns = ['ID', 'Topic', 'Sentiment', 'Tweet']

print(df.head())
print(df['Sentiment'].value_counts())

# -----------------------------
# Clean Text
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['Clean_Tweet'] = df['Tweet'].apply(clean_text)

# -----------------------------
# Sentiment Distribution
# -----------------------------
plt.figure()
sns.countplot(x='Sentiment', data=df)
plt.title("Sentiment Distribution")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# Sentiment by Topic
# -----------------------------
plt.figure()
sns.countplot(x='Topic', hue='Sentiment', data=df)
plt.xticks(rotation=90)
plt.title("Sentiment by Topic")
plt.show()

# -----------------------------
# WordCloud per Sentiment
# -----------------------------
for sentiment in df['Sentiment'].unique():
    text_data = " ".join(df[df['Sentiment'] == sentiment]['Clean_Tweet'])
    wordcloud = WordCloud(width=800, height=400).generate(text_data)

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud - {sentiment}")
    plt.show()

# -----------------------------
# Top 10 Most Common Words
# -----------------------------
all_words = " ".join(df['Clean_Tweet']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(10)

words = [w[0] for w in common_words]
counts = [w[1] for w in common_words]

plt.figure()
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("Top 10 Most Common Words")
plt.show()

print("Sentiment Analysis Completed Successfully.")
