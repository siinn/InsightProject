#!/Users/sche/anaconda/bin/python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from read_data import read_data
import datetime

# set style sheet
plt.style.use("ggplot")
sns.set_style("white")

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


#------------------------------------------------
# start of main function
#------------------------------------------------
if __name__ == "__main__":

    # load list of articles
    dfa = read_data("list_articles")

    # convert string to float
    dfa["claps"] = dfa["claps"].apply(value_to_float)
    dfa["response"] = dfa["response"].apply(value_to_float)

    # remove rows with nan response
    dfa = dfa.dropna(subset=["response"])

    # convert string to datetime
    dfa["post_time"] = pd.to_datetime(dfa['post_time'])

    #------------------------------------------------
    # number of claps
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    
    # plot number of claps
    sns.distplot(dfa["claps"], bins=100, kde=False, ax=ax)
    
    # customize plots
    ax.set_xlabel("Articles")
    ax.set_ylabel("Claps")
    plt.xlim((1, 175000))
    #plt.legend(loc=1)

    # save figure
    plt.savefig("plots/article_claps.pdf")
    
    
    #------------------------------------------------
    # number of response
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    
    # plot number of responses
    sns.distplot(dfa["response"], kde=False, ax=ax)
    
    # customize plots
    ax.set_xlabel("Articles")
    ax.set_ylabel("Response")
    plt.xlim((1, 175))
    #plt.legend(loc=1)

    # save figure
    plt.savefig("plots/article_response.pdf")
    
    
    #------------------------------------------------
    # number of articles as a function of time
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8, 5))

    # new dataframe only containing title and date
    dfb = dfa[["post_time","title"]]
   
    dfb.groupby([dfb["post_time"].dt.year, dfb["post_time"].dt.month]).count()["title"].plot(kind="bar")
    
    # customize plots
    ax.set_xlabel("Time")
    ax.set_ylabel("Articles")
        
    # use tight layout
    plt.tight_layout()

    # save figure
    plt.savefig("plots/article_time.pdf")
    
    

















