import pandas as pd
import numpy as np
from read_data import read_data
import pickle

# load dataset
df = read_data("list_articles")

# retrieve pickled model (maps)
p = open('data/pickle/user_id_map','rb')
user_id_map = pickle.load(p)

p = open('data/pickle/item_id_map','rb')
item_id_map = pickle.load(p)

# retrieve pickled model (warm start)
p = open('data/pickle/prediction_ws','rb')
prediction_ws = pickle.load(p)

# retrieve pickled model (cold start)
p = open('data/pickle/prediction_cs','rb')
prediction_cs = pickle.load(p)

#---------------------------------------------
# function for recommendations (link)
#---------------------------------------------

def get_article_link(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.iteritems():
        if pidx ==  post_index:

            # return link for recommended item
            return  df.loc[df["post_id"] == pid]["link"].iloc[0]

    # return empty string if can't find link
    return ""

def get_article_link_cold(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_cs[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.iteritems():
        if pidx ==  post_index:

            # return link for recommended item
            return  df.loc[df["post_id"] == pid]["link"].iloc[0]

    # return empty string if can't find link
    return ""

#---------------------------------------------
# function for recommendations (title)
#---------------------------------------------

def get_article_title(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.iteritems():
        if pidx ==  post_index:

            # return title for recommended item
            return  df.loc[df["post_id"] == pid]["title"].iloc[0]

    # return empty string if can't find title
    return ""

def get_article_title_cold(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_cs[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.iteritems():
        if pidx ==  post_index:

            # return title for recommended item
            return  df.loc[df["post_id"] == pid]["title"].iloc[0]

    # return empty string if can't find title
    return ""

#---------------------------------------------
# function for recommendations (claps)
#---------------------------------------------

def get_article_claps(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.iteritems():
        if pidx ==  post_index:

            # return claps for recommended item
            return  df.loc[df["post_id"] == pid]["claps"].iloc[0]

    # return empty string if can't find claps
    return ""

def get_article_claps_cold(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_cs[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.iteritems():
        if pidx ==  post_index:

            # return claps for recommended item
            return  df.loc[df["post_id"] == pid]["claps"].iloc[0]

    # return empty string if can't find claps
    return ""


#---------------------------------------------
# function for recommendations (response)
#---------------------------------------------

def get_article_response(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.iteritems():
        if pidx ==  post_index:

            # return response for recommended item
            return  df.loc[df["post_id"] == pid]["response"].iloc[0]

    # return empty string if can't find response
    return ""

def get_article_response_cold(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_cs[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.iteritems():
        if pidx ==  post_index:

            # return response for recommended item
            return  df.loc[df["post_id"] == pid]["response"].iloc[0]

    # return empty string if can't find response
    return ""
