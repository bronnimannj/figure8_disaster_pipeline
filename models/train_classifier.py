# standard libraries
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import pickle

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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
    y = df.drop(columns=['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    
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
    This function returns a sklearn pipeline that process text and then performs
    multi-output classification on the categories in the dataset.

    Args:
        None

    Returns:
        pipeline: a sklearn pipeline model
    """

    pipeline = Pipeline([
        (
            'features', 
            FeatureUnion([
                ('text_pipeline', Pipeline([
                    (
                        'vect', 
                        CountVectorizer(
                            tokenizer = tokenize,
                            max_df=0.75)),
                    ('tfidf', TfidfTransformer(sublinear_tf = False)) ])),
                ('text_length', TextLengthExtractor()) ]
            )),
        (
            'clf', 
            MultiOutputClassifier(
                GradientBoostingClassifier(
                    n_estimators = 500,
                    min_samples_leaf = 1)))
    ])

    return pipeline

def setup_logger(logger_name, log_file, level=logging.INFO):
    """
    This function sets up a new logging file.

    Args:
        logger_name : name of the logger
        log_file    : name of the file where to save it
        level       : base logging level to be used for this logger

    Returns:
        None
    """
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s') # %(asctime)s - %(levelname)s - 
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler) 

def close_logger(logger):
    """
    
    This function closes all handlers on logger object.
    
    Args:
        logger : logger object

    Returns:
        None
    
    """
    if logger is None:
        return
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)    

def evaluate_model(model, X_test, Y_test, category_names, name_experiment):
    """
    This function prints the performance of a pipeline for all the categories that we're trying to predict.
    
    Args:
        model           : The model that we will use to make the predictions with
        X_test          : Attributes that we will use to make the predictions
        Y_test          : True labels for the corresponding attributes in X_test
        category_names  : list of categories that can be predictied
        name_experiment : The name of the experiment and logging file

    Returns:
        None
    """

    y_prediction = model.predict(X_test)

    setup_logger(
        'log_pipeline', 
        f'logs/{name_experiment}_log.txt'
    )
    logger = logging.getLogger('log_pipeline')

    for idx, column in enumerate(Y_test.columns):
        logger.info('feature predicted: ' + column)
        logger.info('-'*53)
        logger.info(
            classification_report(
                Y_test[column],
                y_prediction[:,idx], 
                zero_division = 0
            )
        )
        logger.info("\n")
    close_logger(logger)   

    return None


def save_model(model, model_filepath):
    """
    This function saves the trained model into a specific (pickle) file.

    Args:

        model          : The trained model
        model_filepath : (relative) path where to store the model

    Returns:
        None
    """

    pickle.dump(
        model,
        open(
            model_filepath,
            "wb"
    )
)
    return None


def main():
    """
    This function is the main one that will run the whole pipeline.

    Args:
        None

    Returns:
        None
    """

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
        evaluate_model(
            model, 
            X_test, 
            Y_test, 
            category_names,
            name_experiment = 'second_experiment'
        )

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!') 
    else:
        print(
            """Please provide :
            - the filepath of the disaster messages database as the first argument 
            - the filepath of the pickle file save the model to as the second argument. 
            
            Example: python train_classifier.py ../data/disaster_response.db classifier.pkl""")

if __name__ == '__main__':
    main()
