#!/Users/sche/anaconda/bin/python3
#import csv
import csv
from read_data import read_data

# Prepare tab-delimited data
def prepare_data():

    df = read_data("response")                                          # read raw data
    df["comment"] = 1                                                   # add dummy column representing rating
    columns_title=["user_id","post_id","comment"]                       # rearrange columns
    df=df.reindex(columns=columns_title)
    df.to_csv("data/model_input/df.csv", sep="\t", index=False)         # convert dataframe to tab deliminated file
    return


# Prepare tab-delimited user feature
def prepare_user_feature():

    df = read_data("user_detail_medium")                                # read user dataframe
    dfs = df[["user_id","group_medium"]]                                # only select username and group
    dfs.to_csv("data/model_input/df_user.csv", sep="\t", index=False)   # convert dataframe to tab deliminated file
    return

# return data as DictReader type
def get_data():

    return csv.DictReader((x.decode("utf-8", "ignore") for x in open("data/model_input/df.csv")), delimiter="\t")

# return data as DictReader type
def get_user_feature():

    return csv.DictReader((x.decode("utf-8", "ignore") for x in open("data/model_input/df_user.csv")), delimiter="\t")

#---------------------------
# Read data
#---------------------------
ratings = csv.DictReader((x.decode("utf-8", "ignore").strip() for x in open("testdata/BX-Book-Ratings.csv")), delimiter=";")
book_features = csv.DictReader((x.decode("utf-8", "ignore").strip() for x in open("testdata/BX-Books.csv")), delimiter=";")




for line in islice(book_features, 1):
    print(json.dumps(line, indent=4))




#---------------------------
# Building the ID mappings
#---------------------------
from lightfm.data import Dataset
dataset = Dataset()
#dataset.fit((x['User-ID'] for x in ratings),        # user-id = userid
#            (x["ISBN"] for x in ratings))           # ISBN = item
dataset.fit((x['user_id'] for x in get_data()),        # user-id = userid
            (x['post_id'] for x in get_data()))           # ISBN = item

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
user_features = dataset.build_user_features(((x['user_id'], [x['group_medium']]) for x in get_user_feature()))
print(repr(user_features))






