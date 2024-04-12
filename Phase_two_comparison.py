
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load the dataset
df = pd.read_csv('D:\Sem6\COMP 262 - NLP and Recommender Systems\Project\Final\df_comparison.csv')

print("Initial data exploration:")
print(df.describe())


df['label'].value_counts()


df['reviewText'].head(5)


# Function to calculate VADER sentiment
def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
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


import joblib
vectorizer = joblib.load(r'D:\Sem6\COMP 262 - NLP and Recommender Systems\Project\Final\pickle files\tfidf_vectorizer.pkl')
lr_model = joblib.load(r'D:\Sem6\COMP 262 - NLP and Recommender Systems\Project\Final\pickle files\logistic_regression_model.pkl')
svm_model = joblib.load(r'D:\Sem6\COMP 262 - NLP and Recommender Systems\Project\Final\pickle files\support_vector_machine_model.pkl')
nb_model = joblib.load(r'D:\Sem6\COMP 262 - NLP and Recommender Systems\Project\Final\pickle files\naive_bayes_model.pkl')
gb_model = joblib.load(r'D:\Sem6\COMP 262 - NLP and Recommender Systems\Project\Final\pickle files\gradient_boosting_model.pkl')
mlp_model = joblib.load(r'D:\Sem6\COMP 262 - NLP and Recommender Systems\Project\Final\pickle files\multilayer_perceptron_model.pkl')


vader_sentiment(df['reviewText'][0])


df['vader_sentiment'] = df['reviewText'].astype('U').apply(vader_sentiment)
df['vader_sentiment_label'] = df['vader_sentiment'].apply(vader_sentiment_label)


df['textblob_sentiment'] = df['reviewText'].astype('U').apply(textblob_sentiment)
df['textblob_sentiment_label'] = df['vader_sentiment'].apply(textblob_sentiment_label)


X = vectorizer.transform(df['reviewText'].astype('U'))
y = df['label']


lr_pred = lr_model.predict(X)
svm_pred = svm_model.predict(X)
nb_pred = nb_model.predict(X)
gb_pred = gb_model.predict(X)
mlp_pred = mlp_model.predict(X)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

lr_accuracy = accuracy_score(y, lr_pred)
lr_precision = precision_score(y, lr_pred, average='weighted')
lr_recall = recall_score(y, lr_pred, average='weighted')
lr_f1 = f1_score(y, lr_pred, average='weighted')

svm_accuracy = accuracy_score(y, svm_pred)
svm_precision = precision_score(y, svm_pred, average='weighted')
svm_recall = recall_score(y, svm_pred, average='weighted')
svm_f1 = f1_score(y, svm_pred, average='weighted')

nb_accuracy = accuracy_score(y, nb_pred)
nb_precision = precision_score(y, nb_pred, average='weighted')
nb_recall = recall_score(y, nb_pred, average='weighted')
nb_f1 = f1_score(y, nb_pred, average='weighted')

gb_accuracy = accuracy_score(y, gb_pred)
gb_precision = precision_score(y, gb_pred, average='weighted')
gb_recall = recall_score(y, gb_pred, average='weighted')
gb_f1 = f1_score(y, gb_pred, average='weighted')

mlp_accuracy = accuracy_score(y, mlp_pred)
mlp_precision = precision_score(y, mlp_pred, average='weighted')
mlp_recall = recall_score(y, mlp_pred, average='weighted')
mlp_f1 = f1_score(y, mlp_pred, average='weighted')


txt_accuracy = accuracy_score(y, df['textblob_sentiment_label'])
txt_precision = precision_score(y, df['textblob_sentiment_label'], average='weighted')
txt_recall = recall_score(y, df['textblob_sentiment_label'], average='weighted')
txt_f1 = f1_score(y, df['textblob_sentiment_label'], average='weighted')


vad_accuracy = accuracy_score(y, df['vader_sentiment_label'])
vad_precision = precision_score(y, df['vader_sentiment_label'], average='weighted')
vad_recall = recall_score(y, df['vader_sentiment_label'], average='weighted')
vad_f1 = f1_score(y, df['vader_sentiment_label'], average='weighted')


results = [
    {'Model': 'Logistic Regression', 'Accuracy': lr_accuracy, 'Precision': lr_precision, 'Recall': lr_recall, 'F1 Score': lr_f1},
    {'Model': 'Support Vector Machine', 'Accuracy': svm_accuracy, 'Precision': svm_precision, 'Recall': svm_recall, 'F1 Score': svm_f1},
    {'Model': 'Naive Bayes', 'Accuracy': nb_accuracy, 'Precision': nb_precision, 'Recall': nb_recall, 'F1 Score': nb_f1},
    {'Model': 'Gradient Boosting', 'Accuracy': gb_accuracy, 'Precision': gb_precision, 'Recall': gb_recall, 'F1 Score': gb_f1},
    {'Model': 'Multi-Layer Perceptron', 'Accuracy': mlp_accuracy, 'Precision': mlp_precision, 'Recall': mlp_recall, 'F1 Score': mlp_f1},
    {'Model': 'Textblob', 'Accuracy': txt_accuracy, 'Precision': txt_precision, 'Recall': txt_recall, 'F1 Score': txt_f1},
    {'Model': 'Vader Sentiment', 'Accuracy': vad_accuracy, 'Precision': vad_precision, 'Recall': vad_recall, 'F1 Score': vad_f1}
]


print(tabulate(results, headers='keys', tablefmt='psql'))


