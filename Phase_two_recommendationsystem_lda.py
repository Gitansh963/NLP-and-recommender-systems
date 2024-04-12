
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path = "D:/Sem6/COMP 262 - NLP and Recommender Systems/Project/AMAZON_FASHION.json"

reviews = []
with open(file_path, "r") as file:
    for line in file:
        reviews.append(json.loads(line))
    
df_original = pd.DataFrame.from_dict(reviews)


df_original.columns


df = pd.DataFrame(df_original, columns = ['overall', 'reviewText', 'summary', 'asin'])



df['review'] =  df['reviewText'] + ' ' + df['summary']


df = df [['overall', 'review', 'asin']]


df.shape


df = df.drop_duplicates()


df.shape


import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text


df['review'] = df['review'].astype('U').apply(preprocess_text)


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word not in stop_words]))



df.head()


df = df[df['review'] != ' ']


df.shape

# ## Using LDA approach


asin_value_counts = df['asin'].value_counts()
print(asin_value_counts)



asin_value_counts.head(20)


top_asin = asin_value_counts.head(20).index

selected_reviews = []
for asin in top_asin:
    reviews = df[df['asin'] == asin].head(200)
    selected_reviews.append(reviews)

selected_reviews_df = pd.concat(selected_reviews)



selected_reviews_df.head()


selected_reviews_df['asin'].value_counts()


from nltk.tokenize import word_tokenize
from gensim import corpora, models
text_data = []
for review in selected_reviews_df['review']:
    tokens = word_tokenize(review)
    text_data.append(tokens)


dictionary = corpora.Dictionary(text_data)


corpus = [dictionary.doc2bow(text) for text in text_data]


lda_model = models.LdaModel(corpus=corpus, num_topics=20, id2word=dictionary)


print(lda_model.print_topics())


selected_reviews_df['topics'] = [lda_model.get_document_topics(item) for item in corpus]


selected_reviews_df['topics'].head(2)


def get_dominant_topic(doc_topics):
    return max(doc_topics, key = lambda x: x[1])[0]


selected_reviews_df['dominant_topic'] = selected_reviews_df['topics'].apply(get_dominant_topic)


selected_reviews_df['dominant_topic'].head(2)


# Calculate average rating per topic
topic_avg_ratings = selected_reviews_df.groupby('dominant_topic')['overall'].mean()


topic_avg_ratings



# Apply enhanced rating based on topic
selected_reviews_df['enhanced_rating_lda'] = np.round((selected_reviews_df['overall'] + selected_reviews_df['dominant_topic'].map(topic_avg_ratings))/2, 2)

# Check the enhanced ratings
print("Enhanced ratings:")
print(selected_reviews_df[[ 'overall', 'enhanced_rating_lda','review',]])


# ## Cosine similarity approach using tfidf


selected_reviews_df.head()


selected_reviews_df.columns


selected_reviews_df.shape


selected_reviews_df.review[:5]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer( stop_words='english')
tfidf_matirx = vectorizer.fit_transform(selected_reviews_df['review'])


tfidf_matirx


cosine_similarity = cosine_similarity(tfidf_matirx, tfidf_matirx)


cosine_similarity, cosine_similarity.shape


num_similar_reviews = 5

# Get indices of top similar reviews for each review
top_similar_indices = np.argsort(cosine_similarity, axis=1)[:, -num_similar_reviews:]


similar_ratings = selected_reviews_df['overall'].values[top_similar_indices]


cosine_similarity[:, top_similar_indices]


# Compute the weighted average of ratings based on cosine similarity scores
weighted_ratings = np.sum(similar_ratings * cosine_similarity[:, top_similar_indices], axis=1) / np.sum(cosine_similarity[:, top_similar_indices], axis=1)


weighted_ratings = np.where(np.isnan(weighted_ratings), 0.0, weighted_ratings)


weighted_ratings.shape


weighted_ratings[:5]


weighted_ratings1  = np.amax(weighted_ratings, axis=1)
weighted_ratings1.shape


weighted_ratings1 = np.round(weighted_ratings1, decimals=2)


selected_reviews_df['computed_rating_tfidf'] = weighted_ratings1


print(selected_reviews_df[['overall','review' ,'computed_rating_tfidf']])

# ## Comaprison of Overall , Computed_rating_tfidf, lda approach


comparison_df = selected_reviews_df[['overall', 'review' , 'enhanced_rating_lda', 'computed_rating_tfidf']]

comparison_df.head()


comparison_df = comparison_df.reset_index(drop=True)



ax = comparison_df[:10].plot(kind='line')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


ax = comparison_df[:10].plot(kind='bar')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')





