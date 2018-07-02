import pandas as pd
import numpy as np
from read_data import read_data
import pickle
import json

# load dataset
df = read_data("list_articles")

# retrieve pickled model (maps)
p = open('data/pickle/user_id_map','rb')
user_id_map = pickle.load(p)
p.close()

p = open('data/pickle/item_id_map','rb')
item_id_map = pickle.load(p)
p.close()

p = open('data/pickle/user_feature_names','rb')
user_feature_names = pickle.load(p)
p.close()

# retrieve pickled model (warm start)
p = open('data/pickle/prediction_ws','rb')
prediction_ws = pickle.load(p, encoding='latin1')
p.close()

#---------------------------------------------
# create topic_map to filter by topics
#---------------------------------------------
def post_id_to_post_index(x):

    if x in item_id_map:
        return item_id_map[x]
    else:
        return 0

topic_map = {
"neural-networks": df.loc[df.topic == "neural-networks"].post_id.apply(post_id_to_post_index).tolist(),
"deep-learning": df.loc[df.topic == "deep-learning"].post_id.apply(post_id_to_post_index).tolist(),
"artificial-intelligence": df.loc[df.topic == "artificial-intelligence"].post_id.apply(post_id_to_post_index).tolist(),
"data-science": df.loc[df.topic == "data-science"].post_id.apply(post_id_to_post_index).tolist(),
"machine-learning": df.loc[df.topic == "machine-learning"].post_id.apply(post_id_to_post_index).tolist(),
}

    
#---------------------------------------------
# combined return function
#---------------------------------------------

def get_article_info(input_value, rank, topic):

    # get prediction for input user
    user = input_value
    user_index = user_id_map[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[0][:100]          # top recommendation
   
    # collect all post index from given topics
    post_index_from_all_topics = []
    for single_topic in topic:
        post_index_from_all_topics = post_index_from_all_topics + topic_map[single_topic]

    # filter by topics
    user_top_n_index_filtered = []
    for i, single_post_index in enumerate(user_top_n_index):
        if single_post_index in post_index_from_all_topics:
            user_top_n_index_filtered.append(single_post_index)

    # index of top recommendation
    if len(user_top_n_index_filtered) > rank:
        post_index = user_top_n_index_filtered[rank]                             
    else:
        return {"link":"", "title":"Not available", "claps":0, "response":0}

    # if not empty, retrieve the following from dataframe
    result = {"link":"", "title":"", "claps":0, "response":0}

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map.items():
        if pidx ==  post_index:

            # return link for recommended item
            result["link"] = df.loc[df["post_id"] == pid]["link"].iloc[0]
            result["title"] = df.loc[df["post_id"] == pid]["title"].iloc[0]
            result["claps"] = df.loc[df["post_id"] == pid]["claps"].iloc[0]
            result["response"] = df.loc[df["post_id"] == pid]["response"].iloc[0]

    # return empty string if can't find link
    return json.dumps(result)

