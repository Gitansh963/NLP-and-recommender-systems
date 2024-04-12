
import pandas as pd
import numpy as np
import json
file_path = "AMAZON_FASHION_5.json"

reviews = []
with open(file_path, "r") as file:
    for line in file:
        reviews.append(json.loads(line))

df = pd.DataFrame.from_dict(reviews)


df.head()


df  = df[['overall', 'reviewText']]


selected_reviews = df[df['reviewText'].str.split().str.len() > 100].head(50)



selected_reviews.head()


selected_reviews['reviewLength'] = selected_reviews['reviewText'].apply(lambda x: len(x.split()))



selected_reviews['reviewLength'].describe()


selected_reviews['reviewLength'].hist()


selected_reviews['reviewText'].head(5)


from transformers import pipeline

# Load the pre-trained LLM model for summarization
model_t5 = pipeline("summarization", model="t5-base", tokenizer="t5-base")

# Function to summarize the reviews and convert to 50 words length
def summarize_reviews_t5(reviews):
    summaries = []
    for review in reviews:
        summary = model_t5(review, max_length=50, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries




# Usage
reviews = selected_reviews['reviewText'].astype('U').tolist()
summaries_t5 = summarize_reviews_t5(reviews)



summaries_t5[0]


selected_reviews['summary_t5'] = summaries_t5
selected_reviews['summary_t5_length'] = selected_reviews['summary_t5'].apply(lambda x: len(x.split()))


selected_reviews.head()


from tabulate import tabulate

table_data = selected_reviews[['reviewText', 'summary_t5']].head(5).reset_index(drop=True).values.tolist()
table_headers = ['Index', 'reviewText', 'summary_t5']

print(tabulate(table_data, headers=table_headers, showindex=True))



from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

# Load the pre-trained BART model for summarization
model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-base")

def summarize_reviews_bart(reviews):
    summaries = []
    for review in reviews:
        inputs = tokenizer_bart.encode(review, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model_bart.generate(inputs, max_length=50, min_length=50, num_beams=4, early_stopping=True)
        summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries



summaries_bart = summarize_reviews_bart(reviews)


summaries_bart[0]


selected_reviews['summary_bart'] = summaries_bart
selected_reviews['summary_bart_length'] = selected_reviews['summary_bart'].apply(lambda x: len(x.split()))


selected_reviews.head()


from tabulate import tabulate

table_data = selected_reviews[['reviewText', 'summary_bart']].head(5).reset_index(drop=True).values.tolist()
table_headers = ['Index', 'reviewText', 'summary_bart']

print(tabulate(table_data, headers=table_headers, showindex=True))



from tabulate import tabulate

table_data = selected_reviews[['summary_t5', 'summary_bart']].head(5).reset_index(drop=True).values.tolist()
table_headers = ['Index', 'summary_t5', 'summary_bart']

print(tabulate(table_data, headers=table_headers, showindex=True))


from tabulate import tabulate

table_data = selected_reviews[['reviewLength','summary_t5_length', 'summary_bart_length']].head(5).reset_index(drop=True).values.tolist()
table_headers = ['Index', 'reviewLength','summary_t5_length', 'summary_bart_length']

print(tabulate(table_data, headers=table_headers, showindex=True))


# Part 2
# 


question_reviews = df[df['reviewText'].astype('U').str.contains('\?', case=False)]



question_reviews.shape


question_reviews.drop_duplicates(subset='reviewText', inplace=True)



question_reviews.shape


question_reviews


from transformers import pipeline
from transformers import pipeline

# Load the pre-trained LLM model for text generation
model_llm = pipeline("text-generation", model="gpt2")

# Function to generate response using the LLM model
def generate_response(review):
    prompt = "As an Amazon service representative, answer the following question from a customer."
    response = model_llm(prompt + review, max_length=100, do_sample=True, temperature=0.7)[0]['generated_text']
    return response




# Example usage
reviews_ques = question_reviews['reviewText'].astype('U').tolist()
responses_ans = [generate_response(review) for review in reviews_ques]


question_reviews.shape


len(responses_ans)


def remove_before_newline(string):
    index = string.find("\n\n")
    if index != -1:
        return string[index+2:]
    else:
        return string



responses_ans = list(map(remove_before_newline, responses_ans))



question_reviews['answers'] = responses_ans


print(question_reviews.head())


a = question_reviews['answers']


print(a[210])


print(a[315])


print(a[369])


review_selected = "is this product available in different colors?"

answer = generate_response(review_selected)


answer = remove_before_newline(answer)


answer

