import os, glob
import pandas as pd

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

    # concatenate dataframes and return
    return pd.concat(list_df_s)
