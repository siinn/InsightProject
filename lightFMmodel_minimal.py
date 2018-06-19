#!/Users/sche/anaconda/bin/python3
import csv, ast
from sklearn.feature_extraction import DictVectorizer
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
import numpy as np
import pandas as pd
import pickle


#--------------------------------------
# Options
#--------------------------------------

# get tab-delimited data
def get_data():

    return csv.DictReader((x.decode("utf-8", "ignore") for x in open("data/model_input/df.csv")), delimiter="\t")


if __name__ == "__main__":

    #---------------------------
    # prepare dataset and mapping
    #---------------------------

    # create dataset
    dataset_cs = Dataset(user_identity_features=True)       # cold start

    # create map between user_id, post_id, user_features and internal indices
    dataset_cs.fit((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()))
    
    # print shape
    num_users, num_items = dataset_cs.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
    
    #---------------------------
    # Building the interactions matrix
    #---------------------------
    # create interaction matrix to optimize
    (interactions_cs, weights_cs) = dataset_cs.build_interactions(((x['user_id'], x['post_id']) for x in get_data()))
    print(repr(interactions_cs))

    # split data into train and test dataset
    train_cs, test_cs = random_train_test_split(interactions_cs, test_percentage=0.2, random_state=None)


    #---------------------------
    # train model
    #---------------------------
    model_bpr_cs = LightFM(loss='bpr')          # Bayesian Personalised Ranking model
    model_warp_cs = LightFM(loss='warp')        # Weighted Approximate-Rank Pairwise

    model_bpr_cs.fit(train_cs)
    model_warp_cs.fit(train_cs)

    # additional information about the model
    model_bpr_cs.get_params()
    model_bpr_cs.get_user_representations()

    # retrieve mapping from dataset
    user_id_map_cs, user_feature_map_cs, item_id_map_cs, item_feature_map_cs = dataset_cs.mapping()

    # make predictions for all user
    prediction_bpr_cs = model_bpr_cs.predict_rank(interactions_cs)
    prediction_warp_cs = model_warp_cs.predict_rank(interactions_cs)


    #---------------------------
    # create pickles for production
    #---------------------------
    model_warp_cs.fit(interactions_cs)

    prediction_warp_cs = model_warp_cs.predict_rank(interactions_cs)
    
    # pickle cold start
    f = open('data/pickle/user_id_map_cs','wb')
    pickle.dump(user_id_map_cs,f)
    f.close()
    
    f = open('data/pickle/item_id_map_cs','wb')
    pickle.dump(item_id_map_cs,f)
    f.close()

    f = open('data/pickle/prediction_warp_cs','wb')
    pickle.dump(prediction_warp_cs,f)
    f.close()
    

















