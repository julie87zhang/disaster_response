import sys
import pandas as pd
import numpy as np
import argparse
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    Input: message filepath and categories filepath
    Output: read 2 files and generate merged initial dataframe 
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id',how='left')
    return df


def clean_data(df):
    """
    Input: initial dataframe
    Output: cleansing version of dataframe
    
    """
    categories=df.iloc[:,-1].str.split(";",expand=True)
    row = categories.iloc[0,:].map(lambda x:x[:-2])
    category_colnames = row.tolist()
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
       categories[column] =  categories[column].map(lambda x:x[-1:])
    # convert column from string to numeric
       categories[column] =  categories[column].astype('int')
    # drop the original categories column from `df`
    df=df.drop(columns='categories')
    df = pd.concat([df,categories],axis = 1,join='inner')
    df.drop_duplicates(subset ="message", keep = 'first', inplace = True)
    for column in category_colnames:
        df[column] =  df[column].astype('int')
    return df

def save_data(df, database_filename):
    """
    save clean version of dataframe into database for model training and validation
    
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('process_date_3', engine, index=False) 


def main():
    """
    main function of cleansing part
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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
