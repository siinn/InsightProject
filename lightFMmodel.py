#!/Users/sche/anaconda/bin/python3
import csv, ast
from read_data import read_data
from sklearn.feature_extraction import DictVectorizer
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
import numpy as np
import pandas as pd
import pickle
import random


#--------------------------------------
# Options
#--------------------------------------
# create tab-delimited data
PrepareData = False




# Prepare tab-delimited data
def prepare_data():

    df = read_data("response")                                          # read raw data
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
        df_train.loc[df["user_id"]==user[0], "comment"] = 0
        
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


if __name__ == "__main__":

    # convert raw data to tab delimited format
    if(PrepareData):
        prepare_data()
   

    # perform validation
    if(True):

        # set test group and size (0=all, 3=non-data, 2=data related, 1=data scientist)
        test_fraction = 0.2
        test_group = 1

        # uesr features
        user_features, user_feature_names = get_user_features()
        print(repr(user_features))

        # create dataset
        dataset_cs = Dataset(user_identity_features=True)       # cold start
        dataset_ws = Dataset(user_identity_features=True)       # warm start

        # prepare train and test data by setting rating to 0 for random users
        test_train_split(test_group, test_fraction)

        # create map between user_id, post_id, user_features and internal indices
        #dataset_cs.fit((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()))
        dataset_cs.fit((x['user_id'] for x in get_train_data()),(x['post_id'] for x in get_train_data()))
        dataset_ws.fit((x['user_id'] for x in get_train_data()),(x['post_id'] for x in get_train_data()), user_features=user_features)
        
        # print shape
        num_users, num_items = dataset_cs.interactions_shape()
        print('Num users: {}, num_items {}.'.format(num_users, num_items))
        
        #---------------------------
        # Building the interactions matrix
        #---------------------------
        # create interaction matrix to optimize
        (interactions_cs, weights_cs) = dataset_cs.build_interactions(((x['user_id'], x['post_id']) for x in get_data()))
        (interactions_ws, weights_ws) = dataset_ws.build_interactions(((x['user_id'], x['post_id']) for x in get_data()))
        print(repr(interactions_cs))
        print(repr(interactions_ws))

        # split data into train and test dataset
        train_cs, test_cs = random_train_test_split(interactions_cs, test_percentage=0.2, random_state=None)
        train_ws, test_ws = random_train_test_split(interactions_ws, test_percentage=0.2, random_state=None)


        #---------------------------
        # train model
        #---------------------------
        model_bpr_cs = LightFM(loss='bpr')          # Bayesian Personalised Ranking model
        model_bpr_ws = LightFM(loss='bpr')     
        model_warp_cs = LightFM(loss='warp')        # Weighted Approximate-Rank Pairwise
        model_warp_ws = LightFM(loss='warp')        

        model_bpr_cs.fit(train_cs)
        model_bpr_ws.fit(train_ws, user_features=user_features)
        model_warp_cs.fit(train_cs)
        model_warp_ws.fit(train_ws, user_features=user_features)

        # additional information about the model
        model_bpr_cs.get_params()
        model_bpr_cs.get_user_representations()

        # retrieve mapping from dataset
        user_id_map_cs, user_feature_map_cs, item_id_map_cs, item_feature_map_cs = dataset_cs.mapping()
        user_id_map_ws, user_feature_map_ws, item_id_map_ws, item_feature_map_ws = dataset_ws.mapping()

        # make predictions for all user
        prediction_bpr_cs = model_bpr_cs.predict_rank(interactions_cs)
        prediction_bpr_ws = model_bpr_ws.predict_rank(interactions_ws, user_features=user_features)
        prediction_warp_cs = model_warp_cs.predict_rank(interactions_cs)
        prediction_warp_ws = model_warp_ws.predict_rank(interactions_ws, user_features=user_features)






        #---------------------------
        # create pickles for production
        #---------------------------
        model_warp_cs.fit(interactions_cs)
        model_warp_ws.fit(interactions_ws, user_features=user_features)

        prediction_warp_cs = model_warp_cs.predict_rank(interactions_cs)
        prediction_warp_ws = model_warp_ws.predict_rank(interactions_ws, user_features=user_features)
        
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
        
        # pickle warm start
        f = open('data/pickle/user_id_map_ws','wb')
        pickle.dump(user_id_map_ws,f)
        f.close()
        
        f = open('data/pickle/item_id_map_ws','wb')
        pickle.dump(item_id_map_ws,f)
        f.close()

        f = open('data/pickle/prediction_warp_ws','wb')
        pickle.dump(prediction_warp_ws,f)
        f.close()































    #-------------------------------
    # test
    #-------------------------------

    if (False):
        # Let's say we want prediction for the following user
        user = "46143b2857b6"
        user_index = user_id_map[user]
        user_prediction = np.array(prediction[user_index].todense())
        user_top_n_index = np.argsort(-user_prediction)[:3][0]

        # list to hold top recommended posts
        post_id = []
        post_link = []
        post_title = []

        # dataframe to find link
        dfa = read_data("list_articles")

        for rank in np.arange(3):
            post_index = user_top_n_index[rank]

            # loop over item_id_map, find post_id given index
            for pid, pidx in item_id_map.iteritems():
                if pidx ==  post_index:

                    # get post id
                    post_id.append(pid)

                    # get link and title
                    post_link.append(dfa.loc[dfa["post_id"] == pid]["link"].iloc[0])
                    post_title.append(dfa.loc[dfa["post_id"] == pid]["title"].iloc[0])


    











#    #-----------------------------------------------
#    # Test ground
#    #-----------------------------------------------
#
#    # Set the number of threads; you can increase this
#    # if you have more physical cores available.
#    NUM_THREADS = 2
#    NUM_COMPONENTS = 30
#    NUM_EPOCHS = 3
#    ITEM_ALPHA = 1e-6
#
#    # Define a new model instance
#    model = LightFM(loss='warp',
#                    item_alpha=ITEM_ALPHA,
#                    no_components=NUM_COMPONENTS)
#    
#    # Fit the hybrid model. Note that this time, we pass
#    # in the item features matrix.
#    model = model.fit(train,
#                    user_features=user_features,
#                    epochs=NUM_EPOCHS,
#                    num_threads=NUM_THREADS)
#    
#
#
#
#
#    coo_matrix((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()))
#
#
#
#df = read_data("response")
#
#df = pd.read_csv('csv_file.csv', names=['user_id', 'group_id', 'group_value'])
#df = df.pivot(index='user_id', columns='post_id')
#mat = df.as_matrix())
#
#
#
