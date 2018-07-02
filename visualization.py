#!/Users/sche/anaconda/bin/python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker
from read_data import read_data

# set style sheet
#plt.style.use("ggplot")
sns.set_style("white")
#sns.set_style("darkgrid")



def plot_precision_at_k(val_result, val_result_random, val_result_popular):

    # read dataframe that contains score df = pd.read_csv("data/validation/"+str(val_result))
    df = pd.read_csv("data/validation/"+str(val_result))
    df_random = pd.read_csv("data/validation/"+str(val_result_random))
    df_popular = pd.read_csv("data/validation/"+str(val_result_popular))

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(10, 6))
    
    # plot bedroom counts
    bins = np.arange(0, 0.0008, 0.00005)
    hist_kws={"histtype": "bar", "linewidth": 2,"alpha": 0.5}
    sns.distplot( df["precision_at_k_ws"], bins=bins, ax=ax, kde=False, label='Warm start', color="salmon", hist_kws=hist_kws)
    sns.distplot( df["precision_at_k_cs"], bins=bins, ax=ax, kde=False, label='Cold start', color="dodgerblue", hist_kws=hist_kws)
    sns.distplot( df_random["precision_at_k_random"], bins=bins, ax=ax, kde=False, label='Random', color="gray", hist_kws=hist_kws)
    sns.distplot( df_popular["precision_at_k_mostpopular"], bins=bins, ax=ax, kde=False, label='Most popular', color="orange", hist_kws=hist_kws)

    
    # customize plots
    #ax.set_xlim([0.20,0.30])
    #ax.set_ylim([0,1])
    ax.set_xlabel("Precision at 10", size = 20)
    ax.set_ylabel("Test sample", size = 20)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Precision at 10", size = 20)
    plt.legend(loc=1, prop={'size': 20})

    # save figure
    plt.savefig("plots/precision_at_k_%s.png" % val_result[:-3])
    
    return 


def plot_recall_at_k(val_result, val_result_random, val_result_popular):

    # read dataframe that contains score
    df = pd.read_csv("data/validation/"+str(val_result))
    df_random = pd.read_csv("data/validation/"+str(val_result_random))
    df_popular = pd.read_csv("data/validation/"+str(val_result_popular))

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(10, 6))
    
    # plot bedroom counts
    hist_kws={"histtype": "bar", "linewidth": 2,"alpha": 0.5}
    bins = np.arange(0, 0.008, 0.0005)
    sns.distplot( df["recall_at_k_ws"], bins=bins, ax=ax, kde=False, label='Warm start', color="salmon", hist_kws=hist_kws)
    sns.distplot( df["recall_at_k_cs"], bins=bins, ax=ax, kde=False, label='Cold start', color="dodgerblue", hist_kws=hist_kws)
    sns.distplot( df_random["recall_at_k_random"], bins=bins, ax=ax, kde=False, label='Random', color="gray", hist_kws=hist_kws)
    sns.distplot( df_popular["recall_at_k_mostpopular"], bins=bins, ax=ax, kde=False, label='Most popular', color="orange", hist_kws=hist_kws)
    
    # customize plots
    ax.set_xlabel("Recall at 10", size = 20)
    ax.set_ylabel("Test sample", size = 20)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Recall at 10", size = 20)
    plt.legend(loc=1, prop={'size': 20})

    # save figure
    plt.savefig("plots/recall_at_k_%s.png" % val_result[:-3])
   
    return

def plot_f1_at_k(val_result, val_result_random, val_result_popular):

    # read dataframe that contains score
    df = pd.read_csv("data/validation/"+str(val_result))
    df_random = pd.read_csv("data/validation/"+str(val_result_random))
    df_mostpopular = pd.read_csv("data/validation/"+str(val_result_popular))

    # calculate F1 score
    df["f1_at_k_ws"] = 2*(df["recall_at_k_ws"] * df["precision_at_k_ws"]) / (df["recall_at_k_ws"]+df["precision_at_k_ws"])
    df["f1_at_k_cs"] = 2*(df["recall_at_k_cs"] * df["precision_at_k_cs"]) / (df["recall_at_k_cs"]+df["precision_at_k_cs"])
    df["f1_at_k_random"] = 2*(df_random["recall_at_k_random"] * df_random["precision_at_k_random"]) / (df_random["recall_at_k_random"]+df_random["precision_at_k_random"])
    df["f1_at_k_mostpopular"] = 2*(df_mostpopular["recall_at_k_mostpopular"] * df_mostpopular["precision_at_k_mostpopular"]) / (df_mostpopular["recall_at_k_mostpopular"]+df_mostpopular["precision_at_k_mostpopular"])
    

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(10, 6))
    
    # plot bedroom counts
    hist_kws={"histtype": "bar", "linewidth": 2,"alpha": 0.5}
    bins = np.arange(0, 0.0015, 0.00005)
    sns.distplot( df["f1_at_k_ws"], bins=bins, ax=ax, kde=False, label='Warm start', color="salmon", hist_kws=hist_kws)
    sns.distplot( df["f1_at_k_cs"], bins=bins, ax=ax, kde=False, label='Cold start', color="dodgerblue", hist_kws=hist_kws)
    sns.distplot( df["f1_at_k_random"], bins=bins, ax=ax, kde=False, label='Random', color="gray", hist_kws=hist_kws)
    sns.distplot( df["f1_at_k_mostpopular"], bins=bins, ax=ax, kde=False, label='Most popular', color="orange", hist_kws=hist_kws)
    
    # customize plots
    ax.set_xlabel("F1 score", size = 20)
    ax.set_ylabel("Test sample", size = 20)
    ax.set_xticks(ax.get_xticks()[::2])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("F1 score", size = 20)
    plt.legend(loc=1, prop={'size': 20})

    # save figure
    plt.savefig("plots/f1_at_k_%s.png" % val_result[:-3])
   
    return
    
    
def plot_reciprocal_rank(val_result, val_result_random, val_result_popular):

    # read dataframe that contains score
    df = pd.read_csv("data/validation/"+str(val_result))
    df_random = pd.read_csv("data/validation/"+str(val_result_random))
    df_popular = pd.read_csv("data/validation/"+str(val_result_popular))

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(5, 2.5))
    
    # plot reciprocal rank
    sns.distplot( df["reciprocal_rank_ws"], bins=10, ax=ax, kde=False, label='Warm start')
    sns.distplot( df["reciprocal_rank_cs"], bins=10, ax=ax, kde=False, label='Cold start')
    sns.distplot( df_random["reciprocal_rank_random"], bins=10, ax=ax, kde=False, label='Random')
    sns.distplot( df_popular["reciprocal_rank_mostpopular"], bins=10, ax=ax, kde=False, label='Most popular')
    
    # customize plots
    #ax.set_xlim([0.70,1.0])
    #ax.set_ylim([0,1])
    ax.set_xlabel("Trained model")
    ax.set_ylabel("Reciprocal rank")
    plt.legend(loc=1)

    # save figure
    plt.savefig("plots/reciprocal_rank_%s.png" %val_result[:-3])
   
    return
    
def plot_precision_epoch():

    # read dataframe that contains score
    df = pd.read_csv("data/validation/df.epoch.csv")

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(5, 3))
    
    # plot reciprocal rank
    ax = df.plot()
    
    # customize plots
    #ax.set_xlim([0.70,1.0])
    #ax.set_ylim([0,1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation score")
    plt.legend(loc=2)
    plt.tight_layout()

    # save figure
    plt.savefig("plots/epoch.png")
   
    return 

def plot_most_common_keywords():

    # read dataframe that contains score
    df = pd.read_csv("data/keywords/df.csv")

    # only plot upto top common keywords
    df = df[:20]

    # rename column names
    df = df.rename(index=str, columns={"Unnamed: 0": "keywords", "0": "kwds_count"})
    df["kwds_count"] = df["kwds_count"].apply(lambda x: float(x)/320)
    #df["kwds_count"] = df["kwds_count"].apply(lambda x: x * 2)

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(5, 4))

    # plot keywords with its count as values
    plt.barh(range(len(df)),df.kwds_count.tolist(),tick_label=df.keywords.tolist(), color="red", alpha=0.5)
    
    # customize plots
    ax.set_xlabel("% of users")
    ax.set_ylabel("Tags")
    plt.legend(loc=2)
    plt.gca().invert_yaxis()
    plt.xticks(rotation='vertical')
    plt.tight_layout()

    # save figure
    plt.savefig("plots/keywords.png")

    return 

def plot_articles():

    # load list of articles
    dfa = read_data("list_articles")

    # convert string to datetime
    dfa["post_time"] = pd.to_datetime(dfa['post_time'])

    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8, 5))

    # new dataframe only containing title and date
    dfb = dfa[["post_time","title"]]

    # count number of articles each month
    dfc = dfb.set_index('post_time').resample('M').count()

    # set proper axis
    ax = dfc.plot(x_compat=True, color="red", alpha=0.5)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(8))
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%b\n%y'))
    plt.gcf().autofmt_xdate(rotation=0, ha="center")
    
    # set axis label
    ax.set_xlabel("Time")
    ax.set_ylabel("Articles")

    # remove legend
    ax.legend_.remove()

    # use tight layout
    fig.autofmt_xdate()
    plt.tight_layout()

    # save figure
    plt.savefig("plots/article_time.png")
    
    return 




# main function
if __name__ == "__main__":

    # specify input
    val_result = "df.csv"      # group 0, all users
    val_result_random = "df.random.csv"      # group 2, intermediate users
    val_result_popular = "df.mostpopular.csv"      # group 2, intermediate users

    # plot validation plots
    plot_precision_at_k(val_result, val_result_random, val_result_popular)
    plot_recall_at_k(val_result, val_result_random, val_result_popular)

    # plot F1 score
    plot_f1_at_k(val_result, val_result_random, val_result_popular)

    # plot precision vs epoch
    plot_precision_epoch()

    # plot most common keywords
    plot_most_common_keywords()

    # plot number of articles as a function of time
    plot_articles()



