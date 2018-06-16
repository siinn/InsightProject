#!/Users/sche/anaconda/bin/python3
import re, os, glob, sys, ast, time
import argparse
import pandas as pd
import numpy as np
import json
from datetime import timedelta, date
from surprise import Reader, Dataset, SVD, evaluate
import pickle
#import seaborn as sns
#import matplotlib.pyplot as plt


def read_data(data):

    # list to hold individual data frame
    list_df_s = []

    # retrieve subreddits
    path_data = "data/%s" %data
    list_csv = os.listdir(path_data)

    # loop over subreddits
    for csv in list_csv:

        # retrieve csv file
        path_csv = glob.glob(os.path.join(path_data+"/"+csv))[0]

        # append to dataframe list if not empty
        list_df_s.append(pd.read_csv(path_csv,index_col=None, header=0))


    # concatenate dataframes
    df = pd.concat(list_df_s)

    return df


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n



if __name__ == '__main__':


    #-------------------------------
    # real model
    #-------------------------------

    # Define the format
    reader = Reader(line_format='user item rating', sep='\t')

    # Load the data from the file using the reader format
    data = Dataset.load_from_file('data/users_tab/df.csv', reader=reader)

    # Retrieve the trainset.
    trainset = data.build_full_trainset()

    # fit model
    p_model = open('data/pickle/model','rb')
    model = pickle.load(p_model)

    uid = ""
    iid = ""

    pred = model.predict(uid, iid, r_ui=0, verbose=True)




