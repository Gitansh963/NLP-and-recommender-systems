
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


file_path = "D:/Sem6/COMP 262 - NLP and Recommender Systems/Project/AMAZON_FASHION.json"

reviews = []
with open(file_path, "r") as file:
    for line in file:
        reviews.append(json.loads(line))



df = pd.DataFrame.from_dict(reviews)
df.info()


df.describe()


df.isnull().sum()


review_count = len(df)


average_rating = df['overall'].mean()
average_rating


product_review_counts = df['asin'].value_counts()
product_review_counts


reviews_per_product = df.groupby('asin')['reviewerID'].count()
reviews_per_product


reviews_per_user = df.groupby('reviewerID')['asin'].count()
reviews_per_user


print('-=-=-=-=-=-=-=-Check for null value-=-=-=-=-=-=-=-')
print(df.isnull().sum())



# Drop rows where 'reviewText' columns has NaN values
df = df.dropna(subset=['reviewText'])


# Identify outliers
df['review_length'] = df['reviewText'].apply(lambda x: len(x))
review_length = df['review_length']
review_counts = range(1, len(review_length) + 1)




# Identify the outlier outside 25-75% percentile
review_length_stats = review_length.describe()
review_length_outliers = df[review_length > review_length_stats['75%'] + 1.5 * (review_length_stats['75%'] - review_length_stats['25%'])]

Q1 = review_length_stats['25%']  # 25th percentile
Q3 = review_length_stats['75%']  # 75th percentile
IQR = Q3 - Q1

# Calculate the upper boundary for outliers
upper_bound = Q3 + 1.5 * IQR

# Select rows where review_length is greater than the upper_bound
review_length_outliers = df[df['review_length'] > upper_bound]

review_count = len(df)
product_review_percent = product_review_counts / review_count * 100
less_than_01_percent = product_review_percent < 0.1
index_of_reviews = df[df['asin'].isin(product_review_counts[less_than_01_percent].index)].index
index_of_reviews



# Visualize review lengths with the outliers visible
plt.scatter(review_counts, review_length)
plt.scatter(review_length_outliers.index, review_length_outliers['review_length'], color='red')
plt.xlabel('Review Count')
plt.ylabel('Review Length')
plt.title('Length of Reviews vs Number of Reviews')
plt.show()



plt.plot(review_counts, review_length)
plt.scatter(review_length_outliers.index, review_length_outliers['review_length'], color='red')
plt.scatter(index_of_reviews, df.loc[index_of_reviews, 'review_length'], color='black')
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
      
    text = text.translate(str.maketrans('','', string.punctuation))
    text = re.sub(r'\s+\w{1,3}[\.,;!?]?$', '', text)
    text = text.strip()

    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords]
    
    # perform stemming and lemmatization 
    lemmatizer = WordNetLemmatizer() 
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens] 
    
    # Convert tokens to string
    token_str = ' '.join(lemmatized_tokens)
    
    return token_str


# Function to plot a confusion matrix in a beautiful way
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()



remove_outliers_zscore(df['review_length'])
outliers_boxplot(df['review_length'])


# Box plot for the 'overall' column
plt.figure(figsize=(10, 6))
plt.boxplot(df['overall'], vert=False)
plt.title('Box Plot for Overall Ratings')
plt.xlabel('Ratings')
plt.show()



Q1 = df['overall'].quantile(0.25)
Q3 = df['overall'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['overall'] < lower_bound) | (df['overall'] > upper_bound)]
print("Outliers based on the 'overall' column:")
print(outliers)


# Display word cloud
# show_wordcloud(df['reviewText'])


# Generate a line graph for with the outliers
z_scores = calculate_z_scores(df['review_length'])

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


# Remove all duplicates
df = df.drop_duplicates(subset=['reviewerID', 'asin', 'reviewText'], keep=False)
df.shape
# Remove the outliers
df = df[~df.index.isin(review_length_outliers.index)]
df.shape




overall_counts = df['overall'].value_counts()
distinct_numbers = overall_counts.index

plt.bar(distinct_numbers, overall_counts)
plt.xlabel('Distinct Numbers')
plt.ylabel('Number of Instances')
plt.title('Number of Instances vs Distinct Numbers in Overall Column')
plt.show()



# Label your data based on the value of “rating of the product”
df['label'] = df['overall'].apply(lambda x: 'Positive' if x in [4, 5] else 'Neutral' if x == 3 else 'Negative')


# Remove all the unverified reviews
df = df[df['verified'] == True]
df.shape

label_counts = df['label'].value_counts()
distinct_labels = label_counts.index

plt.bar(distinct_labels, label_counts)
plt.xlabel('Labels')
plt.ylabel('Number of Instances')
plt.title('Number of Instances vs Labels')
plt.show()



# =============================================================================
# Feature extraction
# =============================================================================

df.columns

df.info()

df.isnull().sum()

df = df.drop(columns = ['vote', 'style', 'image','reviewTime','reviewerID','asin','reviewerName','unixReviewTime'])

df.info()

df.isnull().sum()

df = df.drop(columns = ['overall','verified','review_length'])

df.info()

df.isnull().sum()

df['summary'].head(5)

columns_to_consider = ['reviewText', 'label']


# =============================================================================
# =============================================================================



# Keep important columns
df_filtered = pd.DataFrame(df, columns=columns_to_consider)


df_filtered.shape

from nltk.corpus import stopwords
stop = stopwords.words('english')
df_filtered['reviewText'] = df_filtered['reviewText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

stopwords = set(stopwords.words('english'))

df_filtered = df_filtered.drop_duplicates(subset=['reviewText'])

df_filtered.shape




df_filtered['reviewText'] = df_filtered['reviewText'].apply(preprocess_text)


df_filtered['label'].value_counts()

from sklearn.utils import resample

# Balance the dataset
df_balanced = pd.concat([
    resample(df_filtered[df_filtered['label'] == 'Positive'], n_samples=667, random_state=17),
    resample(df_filtered[df_filtered['label'] == 'Neutral'], n_samples=667, random_state=17),
    resample(df_filtered[df_filtered['label'] == 'Negative'], n_samples=666, random_state=17)
])

df_balanced.reset_index(drop=True, inplace=True)


subset_sample = df_filtered.sample(n=2000, random_state=17)

subset_sample.reset_index(drop=True, inplace=True)


# ## with Word2vec


from gensim.models import Word2Vec


# Tokenize the reviews
subset_sample['tokenized_reviews'] = subset_sample['reviewText'].apply(lambda x: x.split())

# Train a Word2Vec model
w2v_model = Word2Vec(subset_sample['tokenized_reviews'], min_count=1)

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    else:
        return np.mean(word2vec_model.wv[doc], axis=0)

# Now we can use the function to create document vectors
X = subset_sample['tokenized_reviews'].apply(lambda x: document_vector(w2v_model, x))


X = np.stack(X)
X = np.nan_to_num(X)
y = subset_sample['label']


X.shape, y.shape


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import joblib



from sklearn.model_selection import StratifiedShuffleSplit
# Create StratifiedShuffleSplit object
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=17)

# Split the data into train and test sets using stratified shuffle split
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



lr = LogisticRegression(random_state=17)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)




svm = SVC(random_state=17)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("Support Vector Machine Accuracy:", svm_accuracy)




nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", nb_accuracy)




gb = GradientBoostingClassifier(random_state=17)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print("Gradient Boosting Accuracy:", gb_accuracy)




mlp = MLPClassifier(random_state=17)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
print("Multilayer Perceptron Accuracy:", mlp_accuracy)

# ## With - tfidf vectorizer


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(subset_sample['reviewText'])

X.shape

y = subset_sample['label']



from sklearn.model_selection import StratifiedShuffleSplit
# Create StratifiedShuffleSplit object
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=17)

# Split the data into train and test sets using stratified shuffle split
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



X_train[0].toarray()


lr = LogisticRegression(random_state=17)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)





svm = SVC(random_state=17)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("Support Vector Machine Accuracy:", svm_accuracy)




nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", nb_accuracy)




gb = GradientBoostingClassifier(random_state=17)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print("Gradient Boosting Accuracy:", gb_accuracy)




mlp = MLPClassifier(random_state=17)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
print("Multilayer Perceptron Accuracy:", mlp_accuracy)

 [markdown]
# ## Saving the Models 





# joblib.dump(lr, 'pickle files/logistic_regression_model.pkl')

# joblib.dump(svm, 'pickle files/support_vector_machine_model.pkl')

# joblib.dump(nb, 'pickle files/naive_bayes_model.pkl')

# joblib.dump(gb, 'pickle files/gradient_boosting_model.pkl')

# joblib.dump(mlp, 'pickle files/multilayer_perceptron_model.pkl')

# joblib.dump(vectorizer, 'pickle files/tfidf_vectorizer.pkl')



 [markdown]
# ## Creating balanced dataset for apple to apple comparison


# df_new = subset_sample

# df_comparison = pd.concat([
#     resample(df_new[df_new['label'] == 'Positive'], n_samples=334, random_state=17),
#     resample(df_new[df_new['label'] == 'Neutral'], n_samples=333, random_state=17),
#     resample(df_new[df_new['label'] == 'Negative'], n_samples=333, random_state=17)
# ])

# df_comparison.reset_index(drop=True, inplace=True)

# df_comparison['label'].value_counts()

# df_comparison.to_csv('df_comparison.csv', index=False)



