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
    
    #---------------------------
    # prepare dataset and user features
    #---------------------------

    # uesr features
    user_features, user_feature_names = get_user_features()

    dataset = Dataset(user_identity_features=True)
    dataset.fit((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()), user_features=user_features)
    #dataset.fit((x['user_id'] for x in get_data()),(x['post_id'] for x in get_data()))
    
    # print shape
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
    
    
    #---------------------------
    # Building the interactions matrix
    #---------------------------
    # create interaction matrix to optimize
    (interactions, weights) = dataset.build_interactions(((x['user_id'], x['post_id']) for x in get_data()))
    print(repr(interactions))
    
    # prepare user features
    #user_features = dataset.build_user_features(((x['user_id'], [x['group_medium']]) for x in get_user_features()))
    print(repr(user_features))
    
   
    #---------------------------
    # build model
    #---------------------------
    model = LightFM(loss='bpr')
    model.fit(interactions, user_features=user_features)
    #model.fit(interactions)

    # additional information about the model
    model.get_params()
    model.get_user_representations()

    # evaluate
    #print("Train precision: %.2f" % precision_at_k(model, interactions, k=5).mean())

    # retrieve mapping from dataset
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

    # make predictions for all user
    prediction = model.predict_rank(interactions, user_features=user_features)

    #---------------------------
    # make prediction for users
    #---------------------------
    
    # pickle user_id_map
    f = open('data/pickle/user_id_map','wb')
    pickle.dump(user_id_map,f)
    f.close()
    
    # pickle item_id_map
    f = open('data/pickle/item_id_map','wb')
    pickle.dump(item_id_map,f)
    f.close()

    # pickle prediction
    f = open('data/pickle/prediction','wb')
    pickle.dump(prediction,f)
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
