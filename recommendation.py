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
#import seaborn as sns
#import matplotlib.pyplot as plt

#------------------------------------------
# arguments
#------------------------------------------
parser = argparse.ArgumentParser()

# add options
parser.add_argument("--google_users", help="Get additional user informations from google Option = 0, 1, 2, ..", nargs=1, type=int)


# parse arguments
args = parser.parse_args()

if args.google_users:
    chunk_index = int(args.google_users[0])

#-------------------------------------------
# options
#-------------------------------------------
CreateCorrelation = False
PlotCorrelation = False
GoogleUsers = args.google_users


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



