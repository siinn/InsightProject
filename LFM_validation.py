#!/Users/sche/anaconda/bin/python3
from __future__ import division
import csv, ast, random
from read_data import read_data
from sklearn.feature_extraction import DictVectorizer
from lightfm.data import Dataset
from lightfm import LightFM, cross_validation, evaluation
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import pickle


#--------------------------------------
# Options
#--------------------------------------
# create tab-delimited data
PrepareData                 = False
RunValidation               = True
RunValidationRandom         = False
RunValidationMostPopular    = False
RunLearningCurve            = False


# Prepare tab-delimited data
def prepare_data():

    df = read_data("response")                                          # read raw data
    df = df.drop_duplicates(subset=["post_id","user_id"])               # remove duplicated entries
    columns_title=["user_id","post_id"]                                 # rearrange columns
    df=df.reindex(columns=columns_title)
    df.to_csv("data/model_input/df.csv", sep="\t", index=False)         # convert dataframe to tab deliminated file
    return

# get tab-delimited data
def get_data():

    return csv.DictReader((x.decode("utf-8", "ignore") for x in open("data/model_input/df.csv")), delimiter="\t")

# get tab-delimited train data
def get_train_data():

    return csv.DictReader((x.decode("utf-8", "ignore") for x in open("data/model_input/df_train.csv")), delimiter="\t")

# get test data
def get_test_data():

    return pd.read_csv("data/model_input/df_test.csv")

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

    user_features  = dv.fit_transform(df["keyword"])
    feature_names = dv.get_feature_names()

    return user_features, feature_names

# train test split
def test_train_split(fraction):

    # user-item data for model
    df_train = pd.read_csv("data/model_input/df.csv", sep='\t') 

    # dataframe with user detail
    df_user_detail = read_data("user_detail_medium")

    # get list of unique user
    unique_users = list(df_user_detail.drop_duplicates(subset="user_id")["user_id"])

    # number of test users
    n_test_users = int(len(unique_users) * fraction)

    # shuffle and select users to drop
    random.shuffle(unique_users)
    df_test_data = pd.DataFrame(unique_users[:n_test_users])
    df_train_users = pd.DataFrame(unique_users[n_test_users:])

    # set rating to 0 for test users
    for index, user in df_test_data.iterrows():
        #df_train.loc[df_train["user_id"]==user[0], "comment"] = 0
        df_train.drop(df_train[df_train["user_id"]==user[0]].index, inplace=True)

    # check before store
    #print(df_train.loc[df_train["user_id"]==df_test_data.iloc[0][0]])
        
    # save training set
    df_train.to_csv("data/model_input/df_train.csv", sep="\t", index=False)
    df_test_data.to_csv("data/model_input/df_test.csv", index=False)
    df_train_users.to_csv("data/model_input/df_train_users.csv", index=False)

    return 
    

# return precision and recall score given prediction and test data
def precision_recall_score(test_data, prediction, user_id_map, item_id_map, k):

    precision_at_k_list = []
    recall_at_k_list = []

    # retrieve post id of prediction for test users
    for index, row in test_data.iterrows():

        #----------------------------
        # retrieve predictions
        #----------------------------
        user_id = row[0]
        user_index = user_id_map[user_id]                                       # index of test users 
        prediction_index_unsorted = np.array(prediction[user_index].todense())  # unsorted prediction of test users
        prediction_index = np.argsort(-prediction_index_unsorted)[0]            # sorted prediction
        prediction_post_id_upto_k_prec = []                                     # list to hold prediction post id
        prediction_post_id_upto_k_rec  = []                                     # list to hold prediction post id

        # predicted post_id corresponding to post_index upto k for precision
        for rank in np.arange(k[0]):
            post_index = prediction_index[rank]                                 # index of k-th predicted post

            # loop over item_id_map, find post_id given index
            for pid, pidx in item_id_map.iteritems():
                if pidx ==  post_index:
                    prediction_post_id_upto_k_prec.append(pid)                  # predicted post id

        # predicted post_id corresponding to post_index upto k for recall
        for rank in np.arange(k[1]):
            post_index = prediction_index[rank]                                 # index of k-th predicted post

            # loop over item_id_map, find post_id given index
            for pid, pidx in item_id_map.iteritems():
                if pidx ==  post_index:
                    prediction_post_id_upto_k_rec.append(pid)                   # predicted post id

        #----------------------------
        # calculate validation metrics
        #----------------------------

        # train data to look up articles test users liked
        df_train = pd.read_csv("data/model_input/df.csv", sep='\t')
        truth_post_id = df_train.loc[df_train["user_id"]==user_id]["post_id"].tolist()

        # common articles between prediction upto k and truth for precision
        match_precision = set(prediction_post_id_upto_k_prec) & set(truth_post_id) 
        match_recall    = set(prediction_post_id_upto_k_rec) & set(truth_post_id) 

        # append the results
        precision_at_k_list.append(len(match_precision) / k[0])
        if (len(truth_post_id) > 0):
            recall_at_k_list.append(len(match_recall) / len(truth_post_id))
        else:
            recall_at_k_list.append(0)

    return precision_at_k_list, recall_at_k_list

# merge scores from multiprocessing
def merge_scores(mp_result):

    # container to hold merged results
    precision_at_k_merged  = []
    recall_at_k_merged     = []

    for mp in mp_result:
        precision_at_k_merged  = precision_at_k_merged + mp[0]
        recall_at_k_merged     = recall_at_k_merged + mp[1]

    return precision_at_k_merged, recall_at_k_merged


def run_validation_random_model(test_fraction, max_val):

    # containers to hold results
    ave_precision_at_k  = []
    ave_recall_at_k     = []
        
    # choose k for precision and recall
    k = [10, 10]

    # entire train data to randomly select item
    df_train = pd.read_csv("data/model_input/df.csv", sep='\t') 

    # list of unique items available
    df_item = df_train["post_id"].drop_duplicates().tolist()

    # perform validation
    validation_itr = 0

    while (validation_itr < max_val):

        print("Start validating random selection, iteration %s" %validation_itr)

        # prevent random failure to abort entire job
        try:

            # count
            validation_itr += 1

            # containers to hold results
            precision_at_k_list = []
            recall_at_k_list = []

            # prepare train and test data by setting rating to 0 for random users
            test_train_split(test_fraction)

            # test data to optain users in test set
            df_test = pd.read_csv("data/model_input/df_test.csv", sep='\t')

            # loop over each test user
            for index, row in df_test.iterrows():

                # user id 
                user_id = row[0]

                # list of item user commented
                truth_post_id = df_train.loc[df_train["user_id"]==user_id]["post_id"].tolist()

                # shuffle item for random recommendation
                random.shuffle(df_item)
                prediction_post_id_upto_k_prec = df_item[:k[0]+1]
                prediction_post_id_upto_k_rec = df_item[:k[1]+1]

                # common articles between prediction upto k and truth for precision
                match_precision = set(prediction_post_id_upto_k_prec) & set(truth_post_id) 
                match_recall    = set(prediction_post_id_upto_k_rec) & set(truth_post_id) 

                # append the results
                # multiplied by test_fraction to simulate known positive in training sample
                precision_at_k_list.append(len(match_precision) / k[0] * (test_fraction))
                if (len(truth_post_id) > 0):
                    recall_at_k_list.append(len(match_recall) / len(truth_post_id) * (test_fraction))
                else:
                    recall_at_k_list.append(0)


            #---------------------------
            # calculate validation score
            #---------------------------
            # append score from each iteration to results
            ave_precision_at_k.append(sum(precision_at_k_list) / len(precision_at_k_list))
            ave_recall_at_k.append(sum(recall_at_k_list) / len(recall_at_k_list))

        except:
            print("teration %s failed. Skipping.." %validation_itr)


    print("Validation score for random selection")
    print(ave_precision_at_k  )
    print(ave_recall_at_k     )

    df_result = pd.DataFrame({
        'precision_at_k_random': ave_precision_at_k,
        'recall_at_k_random': ave_recall_at_k,
        })

    # save to file
    df_result.to_csv("data/validation/df.random.csv", index=False)

    return



def run_validation_mostpopular_model(test_fraction, max_val):

    # containers to hold results
    ave_precision_at_k  = []
    ave_recall_at_k     = []
        
    # choose k for precision and recall
    k = [10, 10]

    # entire train data to the most popular select item
    df_train = pd.read_csv("data/model_input/df.csv", sep='\t') 

    # load list of articles
    dfa = read_data("list_articles")

    # convert string to float
    dfa["claps"] = dfa["claps"].apply(value_to_float)
    dfa["response"] = dfa["response"].apply(value_to_float)

    # remove duplicates
    dfa = dfa.drop_duplicates(subset="post_id", keep="first")

    # sort by number of comments recieved
    dfa = dfa.sort_values(by=["response"], ascending=False)

    # list of unique items available sorted by popularity
    df_item = dfa["post_id"].tolist()

    # perform validation
    validation_itr = 0

    while (validation_itr < max_val):

        print("Start validating most popular, iteration %s" %validation_itr)

        # prevent random failure to abort entire job
        try:

            # count
            validation_itr += 1

            # containers to hold results
            precision_at_k_list = []
            recall_at_k_list = []

            # prepare train and test data by setting rating to 0 for random users
            test_train_split(test_fraction)

            # test data to optain users in test set
            df_test = pd.read_csv("data/model_input/df_test.csv", sep='\t')

            # loop over each test user
            for index, row in df_test.iterrows():

                # user id 
                user_id = row[0]

                # list of item user commented
                truth_post_id = df_train.loc[df_train["user_id"]==user_id]["post_id"].tolist()

                # shuffle item for random recommendation
                prediction_post_id_upto_k_prec = df_item[:k[0]+1]
                prediction_post_id_upto_k_rec = df_item[:k[1]+1]

                # common articles between prediction upto k and truth for precision
                match_precision = set(prediction_post_id_upto_k_prec) & set(truth_post_id) 
                match_recall    = set(prediction_post_id_upto_k_rec) & set(truth_post_id) 

                # append the results
                # multiplied by test_fraction to simulate known positive in training sample
                precision_at_k_list.append(len(match_precision) / k[0] * (test_fraction))
                if (len(truth_post_id) > 0):
                    recall_at_k_list.append(len(match_recall) / len(truth_post_id) * (test_fraction))
                else:
                    recall_at_k_list.append(0)


            #---------------------------
            # calculate validation score
            #---------------------------
            # append score from each iteration to results
            ave_precision_at_k.append(sum(precision_at_k_list) / len(precision_at_k_list))
            ave_recall_at_k.append(sum(recall_at_k_list) / len(recall_at_k_list))

        except:
            print("teration %s failed. Skipping.." %validation_itr)


    print("Validation score for most popular recommendation")
    print(ave_precision_at_k  )
    print(ave_recall_at_k     )

    df_result = pd.DataFrame({
        'precision_at_k_mostpopular': ave_precision_at_k,
        'recall_at_k_mostpopular': ave_recall_at_k,
        })

    # save to file
    df_result.to_csv("data/validation/df.mostpopular.csv", index=False)

    return



def value_to_float(x):

    if type(x) == float or type(x) == int:
        x = str(x)

    if (('K' in x) or ('k' in x)):
        x = x.replace(".","")
        x = x.replace("k","000")
        x = x.replace("K","000")

    try:
        x = float(x)
    except:
        x = 0.0

    return x


def run_validation(test_fraction, max_val):

    # containers to hold results
    ave_precision_at_k_cs   = []
    ave_recall_at_k_cs      = []
    ave_auc_score_cs        = []

    ave_precision_at_k_ws   = []
    ave_recall_at_k_ws      = []
    ave_auc_score_ws        = []
   

    # perform validation
    validation_itr = 0

    while (validation_itr < max_val):

        print("Start validating cold, warm start, iteration %s" %validation_itr)

        # prevent random failure to abort entire job
        try:

            # count
            validation_itr += 1

            # create data_train
            data_cs = Dataset()
            data_ws = Dataset(user_identity_features=True)

            # user featurs
            user_features, user_feature_names = get_user_features()
            print(user_feature_names)

            # create map between user_id, post_id, user_features and internal indices
            data_cs.fit((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()))
            data_ws.fit((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()), user_features=user_features)
            
            # print shape
            num_users, num_items = data_ws.interactions_shape()
            print('Num users: {}, num_items {}.'.format(num_users, num_items))
            
            #---------------------------
            # Building the interactions matrix
            #---------------------------
            # create interaction matrix to optimize
            (interactions_cs, weights_cs) = data_cs.build_interactions(((x['user_id'], x['post_id'])) for x in get_data())
            (interactions_ws, weights_ws) = data_ws.build_interactions(((x['user_id'], x['post_id'])) for x in get_data())
            print(repr(interactions_ws))

            # retrieve mapping from dataset
            user_id_map_cs, user_feature_map_cs, item_id_map_cs, item_feature_map_cs = data_cs.mapping()
            user_id_map_ws, user_feature_map_ws, item_id_map_ws, item_feature_map_ws = data_ws.mapping()

            # split test and train
            interaction_train_cs, interaction_test_cs = cross_validation.random_train_test_split(interactions_cs, test_fraction)
            interaction_train_ws, interaction_test_ws = cross_validation.random_train_test_split(interactions_ws, test_fraction)

            #---------------------------
            # train model
            #---------------------------
            model_cs  = LightFM(learning_rate=0.05, loss='warp')
            model_ws  = LightFM(learning_rate=0.05, loss='warp', no_components=len(user_feature_names))

            model_cs.fit(interaction_train_cs, epochs=30)
            model_ws.fit(interaction_train_ws, user_features=user_features, epochs=30)

            #---------------------------
            # make predictions
            #---------------------------
            precision_at_k_cs = evaluation.precision_at_k(model_cs, interaction_test_cs, interaction_train_cs)
            recall_at_k_cs = evaluation.recall_at_k(model_cs, interaction_test_cs, interaction_train_cs)
            auc_score_cs = evaluation.auc_score(model_cs, interaction_test_cs, interaction_train_cs)

            precision_at_k_ws = evaluation.precision_at_k(model_ws, interaction_test_ws, interaction_train_ws, user_features=user_features)
            recall_at_k_ws = evaluation.recall_at_k(model_ws, interaction_test_ws, interaction_train_ws, user_features=user_features)
            auc_score_ws = evaluation.auc_score(model_ws, interaction_test_ws, interaction_train_ws, user_features=user_features)

            # append score from each iteration to results
            ave_precision_at_k_cs.append(sum(precision_at_k_cs) / len(precision_at_k_cs))
            ave_recall_at_k_cs.append(sum(recall_at_k_cs) / len(recall_at_k_cs))
            ave_auc_score_cs.append(sum(auc_score_cs) / len(auc_score_cs))

            ave_precision_at_k_ws.append(sum(precision_at_k_ws) / len(precision_at_k_ws))
            ave_recall_at_k_ws.append(sum(recall_at_k_ws) / len(recall_at_k_ws))
            ave_auc_score_ws.append(sum(auc_score_ws) / len(auc_score_ws))


        except:
            print("teration %s failed. Skipping.." %validation_itr)


    print("Validation score for test")
    print(ave_precision_at_k_cs  )
    print(ave_recall_at_k_cs     )
    print(ave_auc_score_cs )
    print(ave_precision_at_k_ws  )
    print(ave_recall_at_k_ws     )
    print(ave_auc_score_ws )

    df_result = pd.DataFrame({
        'precision_at_k_cs': ave_precision_at_k_cs,
        'recall_at_k_cs': ave_recall_at_k_cs,
        'auc_score_cs': ave_auc_score_cs,
        'precision_at_k_ws': ave_precision_at_k_ws,
        'recall_at_k_ws': ave_recall_at_k_ws,
        'auc_score_ws': ave_auc_score_ws,
        })

    # save to file
    df_result.to_csv("data/validation/df.csv", index=False)

    return



def run_learning_curve(test_fraction, max_epoch):

    # create data_train
    data  = Dataset(user_identity_features=True)
    
    # user featurs
    user_features, user_feature_names = get_user_features()
    
    # create map between user_id, post_id, user_features and internal indices
    data.fit((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()), user_features=user_features)
    
    # print shape
    num_users, num_items = data.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
    
    #---------------------------
    # Building the interactions matrix
    #---------------------------
    # create interaction matrix to optimize
    (interactions, weights) = data.build_interactions(((x['user_id'], x['post_id'])) for x in get_data())
    print(repr(interactions))
    
    # retrieve mapping from dataset
    user_id_map, user_feature_map, item_id_map, item_feature_map = data.mapping()
    
    # split test and train
    interaction_train, interaction_test = cross_validation.random_train_test_split(interactions, test_fraction)
    
    #---------------------------
    # train model
    #---------------------------
    model_cs  = LightFM(learning_rate=0.05, loss='warp')
    model_ws  = LightFM(learning_rate=0.05, loss='warp', no_components=len(user_feature_names))

    precision_cs = []
    precision_ws = []

    recall_cs = []
    recall_ws = []

    for epoch in range(int(max_epoch/2)):

        model_cs.fit(interaction_train, epochs=int(epoch*2))
        model_ws.fit(interaction_train, user_features=user_features, epochs=int(epoch*2))
   
        # calculate precision and recall for each epoch
        precision_at_k_cs = evaluation.precision_at_k(model_cs, interaction_test, interaction_train)
        precision_at_k_ws = evaluation.precision_at_k(model_ws, interaction_test, interaction_train, user_features=user_features)

        recall_at_k_cs = evaluation.recall_at_k(model_cs, interaction_test, interaction_train)
        recall_at_k_ws = evaluation.recall_at_k(model_ws, interaction_test, interaction_train, user_features=user_features)

        # append to result
        precision_cs.append(sum(precision_at_k_cs) / len(precision_at_k_cs))
        precision_ws.append(sum(precision_at_k_ws) / len(precision_at_k_ws))
        recall_cs.append(sum(recall_at_k_cs) / len(recall_at_k_cs))
        recall_ws.append(sum(recall_at_k_ws) / len(recall_at_k_ws))

    df_result = pd.DataFrame({
        "precision_cs": precision_cs,
        "precision_ws": precision_ws,
        "recall_cs": recall_cs,
        "recall_ws": recall_ws,
        })

    # save to file
    df_result.to_csv("data/validation/df.epoch.csv", index=False)

    return


if __name__ == "__main__":

    # convert raw data to tab delimited format
    if(PrepareData):
        prepare_data()

    if(RunValidation):
        run_validation(0.2, 100)

    if(RunValidationRandom):
        run_validation_random_model(0.2, 100)

    if(RunValidationMostPopular):
        run_validation_mostpopular_model(0.2, 100)
    
    if(RunLearningCurve):
        run_learning_curve(0.2, 80)
