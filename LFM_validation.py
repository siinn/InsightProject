#!/Users/sche/anaconda/bin/python3
from __future__ import division
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


#--------------------------------------
# Options
#--------------------------------------
# create tab-delimited data
PrepareData = False




# Prepare tab-delimited data
def prepare_data():

    df = read_data("response")                                          # read raw data
    df = df.drop_duplicates(subset=["post_id","user_id"])               # remove duplicated entries
    df["comment"] = 1                                                   # add dummy column representing rating
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

# get test data
def get_test_data():

    return dpd.read_csv("data/model_input/df_test.csv")

# train test split
def test_train_split(group, fraction):

    # user-item data for model
    df_train = pd.read_csv("data/model_input/df.csv", sep='\t') 

    # dataframe with user detail
    df_user_detail = read_data("user_detail_medium")

    # select group to drop users from
    if (group ==0):
        test_group = df_user_detail
    else: 
        test_group = df_user_detail.loc[df_user_detail["group_medium"]==group]

    # get list of unique user
    unique_users = list(test_group.drop_duplicates(subset="user_id")["user_id"])

    # number of test users
    n_test_users = int(len(unique_users) * fraction)

    # shuffle and select users to drop
    random.shuffle(unique_users)
    df_test_data = pd.DataFrame(unique_users[:n_test_users])

    # set rating to 0 for test users
    for index, user in df_test_data.iterrows():
        df_train.loc[df_train["user_id"]==user[0], "comment"] = 0

    # check before store
    #print(df_train.loc[df_train["user_id"]==df_test_data.iloc[0][0]])
        
    # save training set
    df_train.to_csv("data/model_input/df_train.csv", sep="\t", index=False)
    df_test_data.to_csv("data/model_input/df_test.csv", index=False)

    return 
    

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


# return precision and recall score given prediction and test data
def precision_recall_score(test_data, prediction, user_id_map, item_id_map, k):

    precision_at_k_list = []
    recall_at_k_list = []
    reciprocal_rank_list = []

    # retrieve post id of prediction for test users
    for index, row in test_data.iterrows():

        #print("Calculating precision at k, recall at k score for user: %s" % row[0])
        user_id = row[0]
        user_index = user_id_map[user_id]                                       # index of test users 
        prediction_index_unsorted = np.array(prediction[user_index].todense())  # unsorted prediction of test users
        prediction_index = np.argsort(-prediction_index_unsorted)[0]            # sorted prediction
        prediction_post_id = []                                                 # list to hold prediction post id

        # predicted post_id corresponding to post_index upto n
        for rank in np.arange(k):
            post_index = prediction_index[rank]                                 # index of k-th predicted post

            # loop over item_id_map, find post_id given index
            for pid, pidx in item_id_map.iteritems():
                if pidx ==  post_index:
                    prediction_post_id.append(pid)                              # predicted post id

        # train data to look up articles test users liked
        df_train = pd.read_csv("data/model_input/df.csv", sep='\t')
        truth_post_id = df_train.loc[df_train["user_id"]==user_id]["post_id"].tolist()

        # common articles between prediction upto k and truth
        match = set(prediction_post_id) & set(truth_post_id) 

        # debugging
        #print("user_id = " + str(user_id))
        #print(prediction_post_id)
        #print(truth_post_id)
        #print("precision = " + str(len(match) / k))

        # find reciprocal rank
        reciprocal_rank_for_user = []
        for i, true_post in enumerate(truth_post_id):
            post_found = False
            for j, pred_post in enumerate(prediction_post_id):
                if (true_post == pred_post):
                    reciprocal_rank_for_user.append(1. / (j+1))     # add to reciprocal rank
                    post_found = True
            if (post_found == False):
                reciprocal_rank_for_user.append(0)                  # reciprocal rank = 0

      
        # average reciprocal rank of this user
        ave_recip = sum(reciprocal_rank_for_user) / len(reciprocal_rank_for_user)

        # append the results
        precision_at_k_list.append(len(match) / k)
        recall_at_k_list.append(len(match) / len(truth_post_id))
        reciprocal_rank_list.append(sum(reciprocal_rank_for_user) / len(reciprocal_rank_for_user))

    return precision_at_k_list, recall_at_k_list, reciprocal_rank_list

# merge scores from multiprocessing
def merge_scores(mp_result):

    # container to hold merged results
    precision_at_k_merged  = []
    recall_at_k_merged     = []
    reciprocal_rank_merged = []

    for mp in mp_result:
        precision_at_k_merged  = precision_at_k_merged + mp[0]
        recall_at_k_merged     = recall_at_k_merged + mp[1]
        reciprocal_rank_merged = reciprocal_rank_merged + mp[2]

    return precision_at_k_merged, recall_at_k_merged, reciprocal_rank_merged



if __name__ == "__main__":

    # convert raw data to tab delimited format
    if(PrepareData):
        prepare_data()
   

    # perform validation
    if(True):

        # set test group and size (0=all, 3=non-data, 2=data related, 1=data scientist)
        test_fraction = 0.2
        test_group = 2

        # uesr features
        user_features, user_feature_names = get_user_features()
        print(repr(user_features))

        # create data_train
        data_train_cs = Dataset()       # cold start
        data_train_ws = Dataset(user_identity_features=True)        # warm start

        # prepare train and test data by setting rating to 0 for random users
        test_train_split(test_group, test_fraction)

        # create map between user_id, post_id, user_features and internal indices
        data_train_cs.fit((x['user_id'] for x in get_train_data()),(x['post_id'] for x in get_train_data()))
        data_train_ws.fit((x['user_id'] for x in get_train_data()),(x['post_id'] for x in get_train_data()), user_features=user_features)
        
        # print shape
        num_users, num_items = data_train_cs.interactions_shape()
        print('Num users: {}, num_items {}.'.format(num_users, num_items))
        
        #---------------------------
        # Building the interactions matrix
        #---------------------------
        # create interaction matrix to optimize
        (interactions_cs, weights_cs) = data_train_cs.build_interactions(((x['user_id'], x['post_id']) for x in get_train_data()))
        (interactions_ws, weights_ws) = data_train_ws.build_interactions(((x['user_id'], x['post_id']) for x in get_train_data()))
        print(repr(interactions_cs))
        print(repr(interactions_ws))

        # retrieve mapping from dataset
        user_id_map_cs, user_feature_map_cs, item_id_map_cs, item_feature_map_cs = data_train_cs.mapping()
        user_id_map_ws, user_feature_map_ws, item_id_map_ws, item_feature_map_ws = data_train_ws.mapping()

        #---------------------------
        # train model
        #---------------------------
        model_bpr_cs = LightFM(learning_rate=0.05, loss='bpr')
        model_bpr_ws = LightFM(learning_rate=0.05, loss='bpr', no_components=15)     
        model_warp_cs = LightFM(learning_rate=0.05, loss='warp')
        model_warp_ws = LightFM(learning_rate=0.05, loss='warp', no_components=15)

        model_bpr_cs.fit(interactions_cs, epochs=10)
        model_bpr_ws.fit(interactions_ws, user_features=user_features, epochs=10)
        model_warp_cs.fit(interactions_cs, epochs=10)
        model_warp_ws.fit(interactions_ws, user_features=user_features, epochs=10)

        # additional information about the model
        #model_warp_ws.get_params()
        #model_warp_ws.get_user_representations()

        #---------------------------
        # make predictions
        #---------------------------

        # make predictions for all user
        #prediction_bpr_cs = model_bpr_cs.predict_rank(interactions_cs)
        #prediction_bpr_ws = model_bpr_ws.predict_rank(interactions_ws, user_features=user_features)
        prediction_warp_cs = model_warp_cs.predict_rank(interactions_cs)
        prediction_warp_ws = model_warp_ws.predict_rank(interactions_ws, user_features=user_features)

        #---------------------------
        # calculate validation score
        #---------------------------
        # choose k
        k = 10

        # cold start
        precision_recall_score_mp_warp_cs = partial( precision_recall_score, prediction = prediction_warp_cs,
                                                    user_id_map = user_id_map_cs, item_id_map = item_id_map_cs, k = k)
        test_data_itr_cs = pd.read_csv('data/model_input/df_test.csv', chunksize=30)         # test data as iterable
        pool_cs          = multiprocessing.Pool(processes=8)                                 # setup pool
        results_warp_cs  = pool_cs.map(precision_recall_score_mp_warp_cs, test_data_itr_cs)  # run MP
        pool_cs.close

        # warm start
        precision_recall_score_mp_warp_ws = partial( precision_recall_score, prediction = prediction_warp_ws,
                                                    user_id_map = user_id_map_ws, item_id_map = item_id_map_ws, k = k)
        test_data_itr_ws = pd.read_csv('data/model_input/df_test.csv', chunksize=30)
        pool_ws          = multiprocessing.Pool(processes=8)
        results_warp_ws  = pool_ws.map(precision_recall_score_mp_warp_ws, test_data_itr_ws)   # run MP
        pool_ws.close

        # merge results from multiprocessing
        precision_at_k_warp_cs, recall_at_k_warp_cs, reciprocal_rank_warp_cs = merge_scores(results_warp_cs)
        precision_at_k_warp_ws, recall_at_k_warp_ws, reciprocal_rank_warp_ws = merge_scores(results_warp_ws)

        print(sum(precision_at_k_warp_cs) / len(precision_at_k_warp_cs))
        print(sum(recall_at_k_warp_cs) / len(recall_at_k_warp_cs))
        print(sum(reciprocal_rank_warp_cs) / len(reciprocal_rank_warp_cs))

        print(sum(precision_at_k_warp_ws) / len(precision_at_k_warp_ws))
        print(sum(recall_at_k_warp_ws) / len(recall_at_k_warp_ws))
        print(sum(reciprocal_rank_warp_ws) / len(reciprocal_rank_warp_ws))


        # cold start
        #test_data = pd.read_csv('data/model_input/df_test.csv', sep="\t")         # test data as iterable
        #precision_recall_score(test_data, prediction = prediction_warp_cs, user_id_map = user_id_map_cs, item_id_map = item_id_map_cs, k = k)
        #precision_recall_score(test_data, prediction = prediction_warp_ws, user_id_map = user_id_map_ws, item_id_map = item_id_map_ws, k = k)

        ## Let's say we want prediction for the following user
        #user = "f5fc2c88a84d"

        #user_index_cs = user_id_map_cs[user]
        #user_index_ws = user_id_map_ws[user]

        #prediction_warp_cs = model_warp_cs.predict_rank(interactions_cs)
        #prediction_warp_ws = model_warp_ws.predict_rank(interactions_ws, user_features=user_features)

        #user_prediction_cs = np.array(prediction_warp_cs[user_index_cs].todense())
        #user_prediction_ws = np.array(prediction_warp_ws[user_index_ws].todense())

        #user_top_n_index_cs = np.argsort(-user_prediction_cs)[:3][0]
        #user_top_n_index_ws = np.argsort(-user_prediction_ws)[:3][0]

        ## list to hold top recommended posts
        #post_id = []
        #post_link = []
        #post_title = []

        ## dataframe to find link
        #dfa = read_data("list_articles")

        #for rank in np.arange(3):
        #    post_index = user_top_n_index[rank]

        #    # loop over item_id_map, find post_id given index
        #    for pid, pidx in item_id_map.iteritems():
        #        if pidx ==  post_index:

        #            # get post id
        #            post_id.append(pid)

        #            # get link and title
        #            post_link.append(dfa.loc[dfa["post_id"] == pid]["link"].iloc[0])
        #            post_title.append(dfa.loc[dfa["post_id"] == pid]["title"].iloc[0])

    
        #precision_recall_score(test_data[2:3], prediction_warp_cs, user_id_map_cs, item_id_map_cs, k)
        #precision_recall_score(test_data[2:3], prediction_warp_ws, user_id_map_ws, item_id_map_ws, k)

        #precision_recall_score(pd.DataFrame(df_train.loc[:1]["user_id"]), prediction_warp_cs, user_id_map_cs, item_id_map_cs, k)
        #precision_recall_score(pd.DataFrame(df_train.loc[:1]["user_id"]), prediction_warp_ws, user_id_map_ws, item_id_map_ws, k)
        
#        mp_result = ThreadPool(processes=4).apply(precision_recall_score, [prediction_warp_cs, test_data, user_id_map_cs, item_id_map_cs, k])
#
#        mp_result = Pool().apply(precision_recall_score_mp, [prediction_warp_cs, test_data, user_id_map_cs, item_id_map_cs, k])
#        #prec_at_k_warp_cs, recall_at_k_warp_cs = Pool().map(precision_recall_score,
#        #                                                    prediction_warp_cs, test_data, user_id_map_cs, item_id_map_cs, k)
#        #prec_at_k_warp_ws, recall_at_k_warp_ws = Pool().map(precision_recall_score,
#        #                                                    prediction_warp_ws, test_data, user_id_map_ws, item_id_map_ws, k)
#
#        #print(sum(prec_at_k_bpr_cs) / len(prec_at_k_bpr_cs))
#        #print(sum(prec_at_k_bpr_ws) / len(prec_at_k_bpr_ws))
#        print(sum(prec_at_k_warp_cs) / len(prec_at_k_warp_cs))
#        print(sum(prec_at_k_warp_ws) / len(prec_at_k_warp_ws))
#
#        #print(sum(recall_at_k_bpr_cs) / len(recall_at_k_bpr_cs))
#        #print(sum(recall_at_k_bpr_ws) / len(recall_at_k_bpr_ws))
#        print(sum(recall_at_k_warp_cs) / len(recall_at_k_warp_cs))
#        print(sum(recall_at_k_warp_ws) / len(recall_at_k_warp_ws))
#
#
#
#
#
#
#        #---------------------------
#        # create pickles for production
#        #---------------------------
#        model_warp_cs.fit(interactions_cs)
#        model_warp_ws.fit(interactions_ws, user_features=user_features)
#
#        prediction_warp_cs = model_warp_cs.predict_rank(interactions_cs)
#        prediction_warp_ws = model_warp_ws.predict_rank(interactions_ws, user_features=user_features)
#        
#        # pickle cold start
#        f = open('data/pickle/user_id_map_cs','wb')
#        pickle.dump(user_id_map_cs,f)
#        f.close()
#        
#        f = open('data/pickle/item_id_map_cs','wb')
#        pickle.dump(item_id_map_cs,f)
#        f.close()
#
#        f = open('data/pickle/prediction_warp_cs','wb')
#        pickle.dump(prediction_warp_cs,f)
#        f.close()
#        
#        # pickle warm start
#        f = open('data/pickle/user_id_map_ws','wb')
#        pickle.dump(user_id_map_ws,f)
#        f.close()
#        
#        f = open('data/pickle/item_id_map_ws','wb')
#        pickle.dump(item_id_map_ws,f)
#        f.close()
#
#        f = open('data/pickle/prediction_warp_ws','wb')
#        pickle.dump(prediction_warp_ws,f)
#        f.close()
#
#
#
#
#
#
#
#
#
#
#
#    #-------------------------------
#    # test
#    #-------------------------------
#
#    if (False):
#        # Let's say we want prediction for the following user
#        user = "46143b2857b6"
#        user_index = user_id_map[user]
#        user_prediction = np.array(prediction[user_index].todense())
#        user_top_n_index = np.argsort(-user_prediction)[:3][0]
#        post_index = user_top_n_index[rank]
#
#        # list to hold top recommended posts
#        post_id = []
#        post_link = []
#        post_title = []
#
#        # dataframe to find link
#        dfa = read_data("list_articles")
#
#        for rank in np.arange(3):
#            post_index = user_top_n_index[rank]
#
#            # loop over item_id_map, find post_id given index
#            for pid, pidx in item_id_map.iteritems():
#                if pidx ==  post_index:
#
#                    # get post id
#                    post_id.append(pid)
#
#                    # get link and title
#                    post_link.append(dfa.loc[dfa["post_id"] == pid]["link"].iloc[0])
#                    post_title.append(dfa.loc[dfa["post_id"] == pid]["title"].iloc[0])
#
#
#    
#
#
#
#
#
#
#
#
#
#
#
##    #-----------------------------------------------
##    # Test ground
##    #-----------------------------------------------
##
##    # Set the number of threads; you can increase this
##    # if you have more physical cores available.
##    NUM_THREADS = 2
##    NUM_COMPONENTS = 30
##    NUM_EPOCHS = 3
##    ITEM_ALPHA = 1e-6
##
##    # Define a new model instance
##    model = LightFM(loss='warp',
##                    item_alpha=ITEM_ALPHA,
##                    no_components=NUM_COMPONENTS)
##    
##    # Fit the hybrid model. Note that this time, we pass
##    # in the item features matrix.
##    model = model.fit(train,
##                    user_features=user_features,
##                    epochs=NUM_EPOCHS,
##                    num_threads=NUM_THREADS)
##    
##
##
##
##
##    coo_matrix((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()))
##
##
##
##df = read_data("response")
##
##df = pd.read_csv('csv_file.csv', names=['user_id', 'group_id', 'group_value'])
##df = df.pivot(index='user_id', columns='post_id')
##mat = df.as_matrix())
##
##
##
