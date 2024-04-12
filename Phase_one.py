
# Import the libraries
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re, string
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns



# Download NLTK packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# # Phase 1
# ## Define functions
# 


# Word cloud
def show_wordcloud(df):
    corpus=[]
    stem = PorterStemmer()
    lem = WordNetLemmatizer()
    for review in df:
        review = str(review)
        words = [w for w in word_tokenize(review) if (w not in stopwords)]

        words = [lem.lemmatize(w) for w in words if len(w)>2]
        corpus.append(words)
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud=wordcloud.generate(str(corpus))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

# Calculate z score
def calculate_z_scores(data):
    return (data - np.mean(data)) / np.std(data)

# Filter out data with z-score more than threshold 3
def remove_outliers_zscore(data, threshold=3):
    z_scores = calculate_z_scores(data)
    filtered_data = data[abs(z_scores) < threshold]
    return filtered_data

# Box plot
def outliers_boxplot(data):
    fig, ax = plt.subplots()
    ax.boxplot(data, showfliers=True)
    # Set showfliers=True to show outliers
    plt.title('Box Plot')
    plt.show()

# Text preprocessing
def preprocess_text(text):
    # Convert the text to lower case
    text = text.lower()
    
    # Remove punctuations, incomplete words, white space, 
    text = text.translate(str.maketrans('','', string.punctuation))
    text = re.sub(r'\s+\w{1,3}[\.,;!?]?$', '', text)
    text = text.strip()
    
    # Tokenize each tweet and remove stop words
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords]
    
    # perform stemming and lemmatization 
    lemmatizer = WordNetLemmatizer() 
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens] 
    
    return lemmatized_tokens

# Function to calculate VADER sentiment
def vader_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

def vader_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to calculate TextBlob sentiment
def textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity

def textblob_sentiment_label(polarity_score):
    if polarity_score > 0:
        return 'Positive'
    elif polarity_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to plot a confusion matrix in a beautiful way
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()


# ## Data Exploration


# Import the dataset
file_path = "AMAZON_FASHION_5.json"

reviews = []
with open(file_path, "r") as file:
    for line in file:
        reviews.append(json.loads(line))

df_group01 = pd.DataFrame.from_dict(reviews)


# Dataframe info
print(df_group01.info())


# Distribution of the number of reviews per products
product_review_counts = df_group01['asin'].value_counts()
product_review_counts


# Distribution of reviews per user
reviews_per_user = df_group01.groupby('reviewerID')['asin'].count()
reviews_per_user


# check for missing values
print(df_group01.isnull().sum())


# check for missing values
print('-=-=-=-=-=-=-=-Check for null value-=-=-=-=-=-=-=-')
print(df_group01.isnull().sum())


# Drop rows where 'reviewText' columns has NaN values
df_group01 = df_group01.dropna(subset=['reviewText'])


# Identify outliers
df_group01['review_length'] = df_group01['reviewText'].apply(lambda x: len(x))
review_length = df_group01['review_length']
review_counts = range(1, len(review_length) + 1)


# Identify the outlier outside 25-75% percentile
review_length_stats = review_length.describe()
review_length_outliers = df_group01[review_length > review_length_stats['75%'] + 1.5 * (review_length_stats['75%'] - review_length_stats['25%'])]

Q1 = review_length_stats['25%']  # 25th percentile
Q3 = review_length_stats['75%']  # 75th percentile
IQR = Q3 - Q1

# Calculate the upper boundary for outliers
upper_bound = Q3 + 1.5 * IQR

# Select rows where review_length is greater than the upper_bound
review_length_outliers = df_group01[df_group01['review_length'] > upper_bound]

review_count = len(df_group01)
product_review_percent = product_review_counts / review_count * 100
less_than_01_percent = product_review_percent < 0.1
index_of_reviews = df_group01[df_group01['asin'].isin(product_review_counts[less_than_01_percent].index)].index
index_of_reviews



# Visualize review lengths with the outliers visible
plt.scatter(review_counts, review_length)
plt.scatter(review_length_outliers.index, review_length_outliers['review_length'], color='red')
plt.xlabel('Review Count')
plt.ylabel('Review Length')
plt.title('Length of Reviews vs Number of Reviews')
plt.show()

plt.scatter(review_counts, review_length)
plt.scatter(review_length_outliers.index, review_length_outliers['review_length'], color='red')
plt.scatter(index_of_reviews, df_group01.loc[index_of_reviews, 'review_length'], color='black')
plt.xlabel('Review Count')
plt.ylabel('Review Length')
plt.title('Length of Reviews vs Number of Reviews')
plt.show()

plt.scatter(review_counts, review_length)
plt.scatter(review_length_outliers.index, review_length_outliers['review_length'], color='red')
plt.xlabel('Review Count')
plt.ylabel('Review Length')
plt.title('Length of Reviews vs Number of Reviews') 
plt.show()


plt.plot(review_counts, review_length)
plt.scatter(review_length_outliers.index, review_length_outliers['review_length'], color='red')
plt.scatter(index_of_reviews, df_group01.loc[index_of_reviews, 'review_length'], color='black')
plt.xlabel('Review Count')
plt.ylabel('Review Length')
plt.title('Length of Reviews vs Number of Reviews')
plt.show()

plt.plot(review_counts, review_length)
plt.scatter(review_length_outliers.index, review_length_outliers['review_length'], color='red')
plt.xlabel('Review Count')
plt.ylabel('Review Length')
plt.title('Length of Reviews vs Number of Reviews') 
plt.show()

# Identify outliers using Z-score and create boxplot
remove_outliers_zscore(df_group01['review_length'])
outliers_boxplot(df_group01['review_length'])


# Box plot for the 'overall' column
plt.figure(figsize=(10, 6))
plt.boxplot(df_group01['overall'], vert=False)
plt.title('Box Plot for Overall Ratings')
plt.xlabel('Ratings')
plt.show()

# Calculate the IQR to identify outliers
Q1 = df_group01['overall'].quantile(0.25)
Q3 = df_group01['overall'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df_group01[(df_group01['overall'] < lower_bound) | (df_group01['overall'] > upper_bound)]
print("Outliers based on the 'overall' column:")
print(outliers)


# Define stopwords
stopwords = set(stopwords.words('english'))

# Display word cloud
show_wordcloud(df_group01['reviewText'])


# Generate a line graph for with the outliers
z_scores = calculate_z_scores(df_group01['review_length'])

plt.figure(figsize=(10, 6))
plt.plot(z_scores, label='Z-scores')
plt.axhline(y=3, color='r', linestyle='--', label='Threshold (+3)')
plt.axhline(y=-3, color='g', linestyle='--', label='Threshold (-3)')
plt.title('Z-scores of Data Points')
plt.xlabel('Data Point Index')
plt.ylabel('Z-score')
plt.legend()
plt.grid(True)
plt.show()


# ## Text pre-processing


# Remove all duplicates
df_group01 = df_group01.drop_duplicates(subset=['reviewerID', 'asin', 'reviewText'], keep=False)

# Remove the outliers
df_group01 = df_group01[~df_group01.index.isin(review_length_outliers.index)]


df_group01.shape


overall_counts = df_group01['overall'].value_counts()
distinct_numbers = overall_counts.index

plt.bar(distinct_numbers, overall_counts)
plt.xlabel('Distinct Numbers')
plt.ylabel('Number of Instances')
plt.title('Number of Instances vs Distinct Numbers in Overall Column')
plt.show()


# Label your data based on the value of “rating of the product”
df_group01['label'] = df_group01['overall'].apply(lambda x: 'Positive' if x in [4, 5] else 'Neutral' if x == 3 else 'Negative')


# Remove all the unverified reviews
df_group01 = df_group01[df_group01['verified'] == True]

label_counts = df_group01['label'].value_counts()
distinct_labels = label_counts.index

plt.bar(distinct_labels, label_counts)
plt.xlabel('Labels')
plt.ylabel('Number of Instances')
plt.title('Number of Instances vs Labels')
plt.show()

# Keep important columns
df_group01_filtered = df_group01[['reviewText','label']]


# Perform pre-processing
df_group01_filtered = df_group01_filtered.copy()
df_group01_filtered['reviewText'] = df_group01_filtered['reviewText'].apply(preprocess_text)


# Remove the duplicate reviews from the pre-processsed dataframe
df_unique_reviews = df_group01_filtered.drop_duplicates(subset=['reviewText'])


# ## Modeling (Sentiment Analysis)


# ### Vader Analysis


# Define variable for sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()
# Convert list of words back to a string for vader and textblob analysis
df_unique_reviews = df_unique_reviews.copy()
df_unique_reviews['reviewText_str'] = df_unique_reviews['reviewText'].apply(lambda x: ' '.join(x))


# Apply vader sentiment analysis and label creation to new columns
df_unique_reviews['vader_sentiment'] = df_unique_reviews['reviewText_str'].apply(vader_sentiment)
df_unique_reviews['vader_sentiment_label'] = df_unique_reviews['vader_sentiment'].apply(vader_sentiment_label)

# Compute accuracy and confusion matrix for vader
vader_accuracy = accuracy_score(df_unique_reviews['label'], df_unique_reviews['vader_sentiment_label'])
vader_confusion_mat = confusion_matrix(df_unique_reviews['label'], df_unique_reviews['vader_sentiment_label'])

# Print accuracy and confusion matrix for vader
print(f'Vader Accuracy: {vader_accuracy}')
print(f'Vader Confusion Matrix: {vader_confusion_mat}')

# Define the labels for the confusion matrix (assuming they are 'Negative', 'Neutral', 'Positive')
class_labels = ['Negative', 'Neutral', 'Positive']

# Plotting the VADER confusion matrix
plot_confusion_matrix(vader_confusion_mat, classes=class_labels, title='VADER Confusion Matrix')

a = vader_sentiment("I do not love this product and it is very good to use and reliable.")
b = vader_sentiment_label(a)
c = textblob_sentiment("I do not love this product and it is very good to use and reliable.")
d =textblob_sentiment_label(c)
# ### TextBlob Analysis
a
b
c
d

# Apply textblob sentiment analysis and label creation to new columns
df_unique_reviews.loc[:, 'textblob_sentiment'] = df_unique_reviews['reviewText_str'].apply(textblob_sentiment)
df_unique_reviews.loc[:, 'textblob_sentiment_label'] = df_unique_reviews['textblob_sentiment'].apply(textblob_sentiment_label)

# Compute accuracy and confusion matrix for textblob
textblob_accuracy = accuracy_score(df_unique_reviews['label'], df_unique_reviews['textblob_sentiment_label'])
textblob_confusion_mat = confusion_matrix(df_unique_reviews['label'], df_unique_reviews['textblob_sentiment_label'])

# Print accuracy and confusion matrix for textblob
print(f'Text Blob Accuracy: {textblob_accuracy}')
print(f'Text Blob Confusion Matrix: {textblob_confusion_mat}')

# Plotting the TextBlob confusion matrix
plot_confusion_matrix(textblob_confusion_mat, classes=class_labels, title='TextBlob Confusion Matrix')


from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('sentiwordnet')
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def sentiwordnet_sentiment(text):
    sentiment = 0.0
    tokens_count = 0
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue 
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1
    if not tokens_count:
        return 0
    return sentiment / tokens_count

df_unique_reviews['sentiwordnet_sentiment'] = df_unique_reviews['reviewText_str'].apply(sentiwordnet_sentiment)
df_unique_reviews['sentiwordnet_sentiment_label'] = df_unique_reviews['sentiwordnet_sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

sentiwordnet_accuracy = accuracy_score(df_unique_reviews['label'], df_unique_reviews['sentiwordnet_sentiment_label'])
sentiwordnet_confusion_mat = confusion_matrix(df_unique_reviews['label'], df_unique_reviews['sentiwordnet_sentiment_label'])

print(f'SentiWordNet Accuracy: {sentiwordnet_accuracy}')
print(f'SentiWordNet Confusion Matrix: {sentiwordnet_confusion_mat}')

plot_confusion_matrix(sentiwordnet_confusion_mat, classes=class_labels, title='SentiWordNet Confusion Matrix')
