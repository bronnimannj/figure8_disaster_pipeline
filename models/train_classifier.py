# standard libraries
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Setting up connection
from sqlalchemy import create_engine

# we will print the logs into a file outside
import logging

# NLP functions
import nltk
# if first time installing nltk: nltk.download()
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# modelling sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix



def load_data(database_filepath):
    """
    This function loads the dataset from database.
    
    It then defines and returns feature and target variable X and Y.
    
    Args:
    database_filepath: path where the database is

    Returns:
    X: features
    y: targets
    """

    engine = create_engine('sqlite:///../data/' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages_with_categories', engine)

    X = df['message']

    # drop columns from df that are not dependent variables
    y = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    
    # remove child_alone as it contains only 1 value
    y.drop(columns=['child_alone'], axis=1, inplace=True)

    return X, y


def tokenize(text):
    """
    This function processes the message
    
    Args:
    text: list of text messages

    Returns:
    clean_tokens: tokenized text
    """

    # First remove punctation and lowercase all letters
    text = re.sub(
        r"[^a-zA-Z0-9]", 
        " ", 
        text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize 
    lemmatizer = WordNetLemmatizer()
    
    # clean the tokens
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """
    Customized class to add the length of text as a feature.
    This class is used in building model
    """

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_length = pd.Series(X).apply(lambda x: len(x))
        return pd.DataFrame(X_length)

def build_model():
    """
    Returns a machine learning pipeline that process text and then performs
    multi-output classification on the categories in the dataset.

    Args:
    None

    Returns:
    a scikit learn pipeline model
    """

    pipeline = Pipeline(
        # ...
    )

    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()


# def train(X, y, model):
#     # train test split


#     # fit model


#     # output model test results


#     return model


# def export_model(model):
#     # Export model as a pickle file



# def run_pipeline(data_file):
#     X, y = load_data(data_file)  # run ETL pipeline
#     model = build_model()  # build model pipeline
#     model = train(X, y, model)  # train model pipeline
#     export_model(model)  # save model


# if __name__ == '__main__':
#     data_file = sys.argv[1]  # get filename of dataset
#     run_pipeline(data_file)  # run data pipeline