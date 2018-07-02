#!/Users/sche/anaconda/bin/python3
from __future__ import division
from datetime import date
import csv, ast, random
from read_data import read_data
from sklearn.feature_extraction import DictVectorizer
from lightfm.data import Dataset
from lightfm import LightFM
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import pickle
from scipy import sparse


#--------------------------------------
# Options
#--------------------------------------
PrepareData     = True
TrainModel      = True
PickleModel     = True

#--------------------------------------


# Prepare tab-delimited data
def prepare_data():

    df = read_data("response")                                          # read raw data
    df = df.drop_duplicates(subset=["post_id","user_id"])               # remove duplicated entries
    columns_title=["user_id","post_id","comment"]                       # rearrange columns
    df=df.reindex(columns=columns_title)
    df.to_csv("data/model_input/df.csv", sep="\t", index=False)         # convert dataframe to tab deliminated file
    return

# get tab-delimited data
def get_data():

    return csv.DictReader((x.decode("utf-8", "ignore") for x in open("data/model_input/df.csv")), delimiter="\t")

# get tab-delimited train data
def get_train_data():

    return csv.DictReader((x.decode("utf-8", "ignore") for x in open("data/model_input/df_train.csv")), delimiter="\t")

# convert list to dictionary
def list_to_dict(keywords):

    # dictionary to return
    result = {}

    # add each key in list to dictionary
    for key in keywords:
        result[key] = 1

    return result

# get sparse matrix of user features
def get_user_features():

    # read user detailed data
    df = read_data("user_detail_medium")

    # drop duplicated users
    df = df.drop_duplicates(subset=["user_id"],keep="first")
    
    # convert string of list to dictionary with equal weight
    df["keyword"] = df["keyword"].apply(lambda x: list_to_dict(ast.literal_eval(x)))
    
    # convert dictionary to sparse matrix of user features
    from sklearn.feature_extraction import DictVectorizer
    dv = DictVectorizer()

    user_features = dv.fit_transform(df["keyword"])
    feature_names = dv.get_feature_names()

    return user_features, feature_names

def train_model():

    # uesr features
    user_features, user_feature_names = get_user_features()
    
    # create data
    data_ws = Dataset(user_identity_features=True)        # warm start
    
    # create map between user_id, post_id, user_features and internal indices
    data_ws.fit((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()), user_features=user_features)
    #user_biases = 
    
    #---------------------------
    # Building the interactions matrix
    #---------------------------
    # create interaction matrix to optimize
    (interactions_ws, weights_ws) = data_ws.build_interactions(((x['user_id'], x['post_id']) for x in get_data()))
    print(repr(interactions_ws))
    
    # retrieve mapping from dataset
    user_id_map, user_feature_map, item_id_map, item_feature_map = data_ws.mapping()
    
    #---------------------------
    # train model
    #---------------------------
    # initialize model
    model_warp_ws = LightFM(learning_rate=0.05, loss='warp', no_components=len(user_feature_names))

    # train model
    model_warp_ws.fit(interactions_ws, user_features=user_features, epochs=30)

    #---------------------------
    # make predictions
    #---------------------------
    # make predictions for all user
    prediction_ws = model_warp_ws.predict_rank(interactions_ws, user_features=user_features)

    # create identity matrix that represent user features of hypothetical user
    user_features_identity = sparse.csr_matrix(np.identity(len(user_feature_names)))

    # make prediction for hypothetical user
    prediction_hypo = []

    for user_irt in range(len(user_feature_names)):

        # calculate prediction score
        prediction_score = model_warp_ws.predict(user_ids=0, item_ids=item_id_map.values(), user_features=user_features_identity)

        # combine prediction score with item map
        prediction_zipped = zip(prediction_score, item_id_map)

        # sort by prediction score
        prediction_sorted = sorted(prediction_zipped, key=lambda x: x[0], reverse =True)

        # add to list of hypothetical users
        prediction_hypo.append(prediction_sorted)


    return prediction_hypo, prediction_ws, user_id_map, item_id_map

def pickle_model(prediction_hypo, prediction_ws, user_id_map, item_id_map):

        # pickle user, item to id map
        f = open('data/pickle/user_id_map','wb')
        pickle.dump(user_id_map,f)
        f.close()
        
        f = open('data/pickle/item_id_map','wb')
        pickle.dump(item_id_map,f)
        f.close()
        
        # pickle warm start
        f = open('data/pickle/prediction_ws','wb')
        pickle.dump(prediction_ws,f)
        f.close()

        # pickle hypothetical users
        f = open('data/pickle/prediction_hypo','wb')
        pickle.dump(prediction_hypo,f)
        f.close()

# main function
if __name__ == "__main__":

    # convert raw data to tab delimited format
    if(PrepareData):
        prepare_data()

    # train model and make predictions
    if(TrainModel):
        prediction_hypo, prediction_ws, user_id_map, item_id_map = train_model()

    # pickle prediction for production
    if(PickleModel):
        pickle_model(prediction_hypo, prediction_ws, user_id_map, item_id_map)

    # write to log for pipeline monitoring
    with open("log", "a") as log:
        log.write("\nStep 5. Successfully built a new model")

    with open("log_model_update", "a") as log:
        log.write("\nLast update: %s" % str(date.today()))

