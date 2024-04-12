# Sentiment Analysis Project

This project, hosted on GitHub, involves a comprehensive sentiment analysis using both lexicon-based and machine learning approaches. The project is divided into two main phases, each contained in its own file.

## Phase 1 (`Phase_one.py`)

The first phase involves an in-depth exploration of the dataset, including counts, averages, distribution of reviews across products, and checking for duplicates. The data is labeled based on the product rating and the appropriate columns for the sentiment analyzer are chosen. Three lexicon packages (VADR, TextBlob, SENTIWORDNET) are studied and two are chosen for model building. The text is pre-processed as needed for each model. Two sentiment analysis models are built using the labeled pre-processed data for both the lexicon packages. The results of both models are validated and a comparison table is provided.

## Phase 2 (`Phase_two.py`)

The second phase involves selecting a subset of the original data and carrying out data exploration and pre-processing. The text is represented using one of the text representations discussed in the course. Two sentiment analysis models are built using 70% of the data. The two models are tested using the 30% test data and the accuracy, precision, recall, confusion matrix, and F1 score are noted in the report.

## Phase 2 Comparison (`Phase_two_comparison.py`)

An experiment is designed to compare the test results of the Lexicon model versus the two machine learning models. Both models are run on the same data and the results are compared using appropriate metrics.

## Phase 2 Recommender Systems (`Phase_two_recommendationsystem_lda.py`)

Different techniques of recommender systems are implemented to enhance the rating values of the data using the review data.

## Phase 2 LLM (`Phase_two_llm.py`)

10 reviews with lengths more than 100 words are selected, and using a LLM model, the results are summarized into a 50 word. One review that carries a question nature is selected, and using a LLM model, a response is automatically created as if it were from a service representative.

This project includes a project report, and documented code.
