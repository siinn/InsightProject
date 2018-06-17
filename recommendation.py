#!/Users/sche/anaconda/bin/python3
import re, os, glob, sys, ast, time
import argparse
import pandas as pd
import numpy as np
import json
from datetime import timedelta, date
from surprise import Reader, Dataset, SVD, evaluate
import pickle
from collections import defaultdict
from read_data import read_data


#-------------------------------------------
# options
#-------------------------------------------


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

    # prepare tabbed data?
    if (False):

        # read raw data
        df = read_data("users")

        # add dummy column representing rating
        df["comment"] = 1

        # rearrange columns
        columns_title=["user_id","post_id","comment"]
        df=df.reindex(columns=columns_title)

        # convert dataframe to tab deliminated file
        df.to_csv("data/users_tab/df.csv", sep="\t", index=False, header=False)


    # Define the format
    reader = Reader(line_format='user item rating', sep='\t')

    # Load the data from the file using the reader format
    data = Dataset.load_from_file('data/users_tab/df.csv', reader=reader)

    # split n-fold
    data.split(n_folds=5)

    # initialize model
    model = SVD()
    evaluate(model, data, measures=['RMSE', 'MAE'])

    # Retrieve the trainset.
    trainset = data.build_full_trainset()

    # fit model
    model.fit(trainset)

    # create test dataset
    testset = trainset.build_anti_testset()
    predictions = model.test(testset)

    # find top n for users
    top_n = get_top_n(predictions, n=6)
    
    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])


    #pred = model.predict(uid, iid, r_ui=0, verbose=True)

    # pickle model
    import pickle

    # pickle top_n
    f_top_n = open('data/pickle/top_n','wb')
    pickle.dump(top_n,f_top_n)
    f_top_n.close()

    # pickle model
    f_model = open('data/pickle/model','wb')
    pickle.dump(model,f_model)
    f_model.close()

    # pickle prediction
    f_predictions = open('data/pickle/predictions','wb')
    pickle.dump(predictions,f_predictions)
    f_predictions.close()



