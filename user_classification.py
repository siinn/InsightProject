#!/Users/sche/anaconda/bin/python3
import re, os, glob, sys, ast, time
import argparse
import pandas as pd
import numpy as np
import requests
import json
from datetime import timedelta, date
from googleapiclient.discovery import build
import seaborn as sns
import matplotlib.pyplot as plt
from read_data import read_data

#------------------------------------------
# arguments
#------------------------------------------
parser = argparse.ArgumentParser()

# add options
parser.add_argument("--google_users", help="Get additional user informations from google Option = 0, 1, 2, ..", nargs=1, type=int)


# parse arguments
args = parser.parse_args()

if args.google_users:
    chunk_index = int(args.google_users[0])

#-------------------------------------------
# options
#-------------------------------------------
CreateCorrelation = False
PlotCorrelation = False
GoogleUsers = args.google_users

# google API
google_api_key = ""
google_cse_id = ""



def clean_json_response(response):
    return json.loads(response.text.split('])}while(1);</x>')[1])


def decorate_group(df, method):

    # choose column to look up
    if (method == "medium"):

        user_detail = "bio"

        # convert uesr info from string to dict
        df["userinfo"] = df.userinfo.apply(ast.literal_eval)

        # convert dict to columns
        df = pd.concat([df.drop(['userinfo'], axis=1), df['userinfo'].apply(pd.Series)], axis=1)

    if (method == "google"):
        user_detail = "google_search"

    # add column to specify uer group. By default, all user belongs to group 3
    df["group_%s" % method] = 3

    # classfy users by bio
    group1a = "Data scientist|data scientist|Data Scientist"
    group2a = "Data science|Data Science|data science"
    group2b = "machine learning|Machine learning|Machine Learning"
    group2c = "engineering|engineer|Engineering|Engineer"
    group2d = "student|Student|candidate|Candidate|Amateur|amateur" 


    # group 2
    df.loc[df[user_detail].str.contains(group2a), ["group_%s" % method]] = 2
    df.loc[df[user_detail].str.contains(group2b), ["group_%s" % method]] = 2
    df.loc[df[user_detail].str.contains(group2c), ["group_%s" % method]] = 2
    df.loc[df[user_detail].str.contains(group2d), ["group_%s" % method]] = 2


    # group 1
    df.loc[df[user_detail].str.contains(group1a), ["group_%s" % method]] = 1

    # sub dataframe for each group
    df1 = df.loc[df['group_%s' % method]==1]
    df2 = df.loc[df['group_%s' % method]==2]
    df3 = df.loc[df['group_%s' % method]==3]

    # sort by group number
    df = df.sort_values(by=['group_%s' % method])

    return df

def create_correlation(df):

    # read dataframe containing detailed user information
    df = read_data("user_detail")
    
    # decorate df with groups obtained from medium profile
    # !!!!!!!!!!!!!! can be changed to google profile
    df = decorate_group(df, "medium")

    # create correlation matrix
    mat_corr= np.zeros((len(df), len(df)))

    # fill correlation matrix by comparing commented artices
    for i, post_id_i in enumerate(df["post_id"]):
        for j, post_id_j in enumerate(df["post_id"]):

            mat_corr[i,j] = len(set(post_id_i.split()) & set(post_id_j.split()))


    # convert matrix to dataframe
    df_corr = pd.DataFrame(mat_corr)

    # save dataframe
    df_corr.to_csv("data/user_corr/df_corr.csv", encoding='utf-8', index=False, compression='gzip')

    return

def google_users():

    # read dataframe containing detailed user information
    df = read_data("user_detail")

    # chunk size
    n = 1000
    
    # split dataframe into subset
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
    
    print("Getting user information using Google for dataframe %s / %s" % (chunk_index+1, len(list_df)))
    
    # sub dataframe of list of articles
    df_sub = list_df[chunk_index]
    
    # decorate df with groups obtained from medium profile
    df_sub = decorate_group(df_sub, "medium")
    
    # retrieve user information from google
    df_sub["google_search"] = df_sub["name"].apply(google_search)
    
    # save to csv
    df_sub.to_csv("data/user_detail_googled/users_googled_%s.csv" % chunk_index, encoding='utf-8', index=False)

    print("Successfully retrieved user information using Google for dataframe %s / %s" % (chunk_index+1, len(list_df)+1))

    return

def plot_correlation():

    # read correlation dataframe
    #df_corr = pd.read_csv("data/user_corr/df_corr.csv", index_col=None, header=0, compression='gzip')
    df_corr = pd.read_csv("data/user_corr/df_corr_10000.csv", index_col=None, header=0, compression='gzip')

    #reduce size
    df_corr = df_corr.iloc[:4000,:4000]

    # set style sheet
    plt.style.use("ggplot")
    sns.set_style("white")
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8, 8))
    
    # plot bedroom counts
    ax = sns.heatmap(df_corr, annot=True, fmt=".0f")
    
    # customize plots
    ax.set_xlabel("Users")
    ax.set_ylabel("Users")

    # save figure
    plt.savefig("plots/correlation_matrix.png")

    return

def google_search(search_terms):

    search_results = []

    for search_term in search_terms:

        # define search term
        search_term = '"Alessandro La Placa"'
        search_term = search_term + " data science"

        # call search API
        service = build("customsearch", "v1", developerKey=google_api_key)
        res = service.cse().list(q=search_term, cx=google_cse_id).execute()

        # empty string to hold result
        result = ""

        # concatenate results
        for item in res["items"]:
            #result = result + " " + item["snippet"]
            result = result + " " + item["title"]

        search_results.append(result)

    return search_results



if __name__ == '__main__':

    # create correlation matrix (takes long time)
    if (CreateCorrelation):
        create_correlation(df)

    # plot correlation matrix
    if (PlotCorrelation):
        plot_correlation()

    if (GoogleUsers):
        google_users()

