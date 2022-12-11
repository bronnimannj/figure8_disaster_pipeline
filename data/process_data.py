import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads data sets, and inner-merge them.
    Note: we remove the duplicates

    Args:
    message_filepath: filepath for message.csv
    categories_filepath: filepath for categories.csv
    
    Returns:
    df: the merged data frame.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    messages.drop_duplicates(
        subset = ['id'],
        keep = 'last',
        inplace=True
    )
    categories.drop_duplicates(
        subset = ['id'],
        keep = 'last',
        inplace=True
    )

    df = pd.merge(
        messages,
        categories,
        on='id',
        how='inner')

    return df


def clean_data(df):
    """
    This function cleans the data.

    Args:
    df: the data to clean
    
    Returns:
    df: the cleaned data
    """
    
    categories = df['categories'].str.split(pat=';', expand=True)
    
    row = categories.head(1)

    category_colnames = row.apply(lambda x: x.str[:-2])
    
    categories.rename(columns=category_colnames.iloc[0], inplace=True)

    for column in categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x[-1:]))
    

    categories['related'].replace(2, 1, inplace=True)

    df.drop(columns=['categories'], axis=0, inplace=True)
    df = pd.concat([df, categories], axis=1)

    return df


def save_data(df, database_filename):
    """
    This function stores df in a SQLite database in the specified database file path (database_filename)
    
    Args:
    df: the data to save
    database_filepath: path were database file is saved
    
    Returns:
    None
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_with_categories', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()