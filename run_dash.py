import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pickle
import os, glob
import pandas as pd
import numpy as np
import base64
from read_data import read_data


# load dataset
df = read_data("list_articles")
df_user = read_data("user_detail_medium")

# load pickles

# retrieve pickled model (warm start)
p = open('data/pickle/user_id_map_ws','rb')
user_id_map_ws = pickle.load(p)

p = open('data/pickle/item_id_map_ws','rb')
item_id_map_ws = pickle.load(p)

p = open('data/pickle/prediction_warp_ws','rb')
prediction_ws = pickle.load(p)

# retrieve pickled model (cold start)
p = open('data/pickle/user_id_map_cs','rb')
user_id_map_cs = pickle.load(p)

p = open('data/pickle/item_id_map_cs','rb')
item_id_map_cs = pickle.load(p)

p = open('data/pickle/prediction_warp_cs','rb')
prediction_cs = pickle.load(p)

# create app
app = dash.Dash(__name__, static_folder='static')

# goverment/health theme
app.css.append_css({'external_url': 'https://codepen.io/plotly/pen/EQZeaW.css'})

# custoom local image
image_filename = 'static/analytics_g.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# create layout
app.layout = html.Div([
    
    html.Hr(style={'padding': '0px 0px 0px 0px','margin': '0px 0px 50px 0px'}),

    # container
    html.Div([

        # user selection
        html.Div([

            # logo
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width="100px", style={'display':'block'}),
                html.H1('Medium', style={'margin': '0px 0px 0px 0px'}),
                html.H6('Improved recommendation system'),
            ], style={'width': '300px','display': 'inline-block', 'padding': '0px 0px 0px 0px'}),

            # drop dropdown and multi-select dropdown
            html.Div([
                html.Label('Select user'),
                dcc.Dropdown(
                    id='selected_user',
                    options=[
                        {'label': 'User1', 'value': 'ed47957c7bbd'},
                        {'label': 'User2', 'value': '78ff5a2855cb'},
                        {'label': 'User3', 'value': '8a5d3f4a4655'},
                        {'label': 'User4', 'value': '29d86acdab2b'},
                        {'label': 'User5', 'value': 'd3e12cffc59a'}
                    ],
                    value='ed47957c7bbd',
                ),
            ], style={'display': 'block','padding': '10px 0px 10x 0px'}),

            html.Div([
                html.Label('What\'s your interest?'),
                dcc.Dropdown(
                    options=[
                        {'label': 'Data science', 'value': 'ds'},
                        {'label': 'Machine learning', 'value': 'ml'},
                        {'label': 'AI', 'value': 'ai'}
                    ],
                    value=['ds', 'ml'],
                    multi=True
                ),
            ], style={'display': 'block','padding': '10px 0px 10px 0px'}),

            # custom user name and slider bar
            html.Div([

                html.Label('Name'),
                dcc.Input(id='custom_user', value='Siinn Che', type='text'),
            ], style={'display': 'block','padding': '0px 0px 10px 0px'}),

            html.Div([

                html.Label('Slider'),
                dcc.Slider(
                    min=0,
                    max=3,
                    marks={0:"All", 1:"Non-data related", 2:"Data related", 3:"Data scientist"},
                    value=0
                ),
            ], style={'padding': '0px 20px 80px 0px'}),

            html.P(id='usergroup'),


        ], style={'width': '300px','display': 'inline-block', "float":"left",'padding': '0px 25px 0px 0px'}),


        # print user information
        html.Div([
            html.H2(id='username'),
        ], style={'width': '1000px', 'display': 'inline-block', 'vertical-align': 'top','padding': '50px 0px 0px 50px'}),


        # recommendation output
        html.Div([

            html.H3('Recommendation (Cold start)'),
            html.Li(html.A(id='t1_pred_link_cold', target="_blank", children="dummy", style={"text-decoration": "none"})),
            html.Li(html.A(id='t2_pred_link_cold', target="_blank", children="dummy", style={"text-decoration": "none"})),
            html.Li(html.A(id='t3_pred_link_cold', target="_blank", children="dummy", style={"text-decoration": "none"})),


        ], style={'width': '500px', 'display': 'inline-block', 'vertical-align': 'top','padding': '30px 0px 0px 50px'}),

        html.Div([

            html.H3('Recommendation (Improved)'),
            html.Li(html.A(id='t1_pred_link', target="_blank", children="dummy", style={"text-decoration": "none"})),
            html.Li(html.A(id='t2_pred_link', target="_blank", children="dummy", style={"text-decoration": "none"})),
            html.Li(html.A(id='t3_pred_link', target="_blank", children="dummy", style={"text-decoration": "none"})),

        ], style={'width': '500px', 'display': 'inline-block', 'vertical-align': 'top','padding': '30px 0px 0px 50px'})

    ], style={"max-width":"1600px",'width': '90%', "position":"relative", "float":"left", "margin":"0 auto", 'padding': '0px 50px 0px 50px', "box-sizing":"border-box"}),
    #], className="container"),
])



@app.callback(
    Output(component_id='username', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def update_username(input_value):
    name = df_user.loc[df_user["user_id"]==input_value, ["name"]].iloc[0][0]
    return 'Hi, {}! Here are some interesting articles to read.'.format(name)




@app.callback(
    Output(component_id='usergroup', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def update_usergroup(input_value):
    name = df_user.loc[df_user["user_id"]==input_value, ["group_medium"]].iloc[0][0]
    return 'User group: "{}"'.format(name)



#            post_title.append(df.loc[df["post_id"] == pid]["title"].iloc[0])
#                post_id.append(pid)


def get_article_link(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map_ws[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[:3][0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map_ws.iteritems():
        if pidx ==  post_index:

            # return link for recommended item
            return  df.loc[df["post_id"] == pid]["link"].iloc[0]

    # return empty string if can't find link
    return ""

def get_article_link_cold(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map_cs[user]                                  # internal index of user
    user_prediction = np.array(prediction_cs[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[:3][0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map_cs.iteritems():
        if pidx ==  post_index:

            # return link for recommended item
            return  df.loc[df["post_id"] == pid]["link"].iloc[0]

    # return empty string if can't find link
    return ""

@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='href'),
    [Input(component_id='selected_user', component_property='value')]
)
def t1_pred_link(input_value):

    # return article link with rank 1
    return get_article_link(input_value, 1)

@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link_cold', component_property='href'),
    [Input(component_id='selected_user', component_property='value')]
)
def t1_pred_link_cold(input_value):

    # return article link with rank 1
    return get_article_link_cold(input_value, 1)


@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='href'),
    [Input(component_id='selected_user', component_property='value')]
)
def t2_pred_link(input_value):

    # return article link with rank 2
    return get_article_link(input_value, 2)

@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link_cold', component_property='href'),
    [Input(component_id='selected_user', component_property='value')]
)
def t2_pred_link_cold(input_value):

    # return article link with rank 2
    return get_article_link_cold(input_value, 2)

@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='href'),
    [Input(component_id='selected_user', component_property='value')]
)
def t3_pred_link(input_value):

    # return article link with rank 3
    return get_article_link(input_value, 3)

@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link_cold', component_property='href'),
    [Input(component_id='selected_user', component_property='value')]
)
def t3_pred_link_cold(input_value):

    # return article link with rank 3
    return get_article_link_cold(input_value, 3)



def get_article_title(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map_ws[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[:3][0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map_ws.iteritems():
        if pidx ==  post_index:

            # return title for recommended item
            return  df.loc[df["post_id"] == pid]["title"].iloc[0]

    # return empty string if can't find title
    return ""

def get_article_title_cold(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map_cs[user]                                  # internal index of user
    user_prediction = np.array(prediction_cs[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[:3][0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map_cs.iteritems():
        if pidx ==  post_index:

            # return title for recommended item
            return  df.loc[df["post_id"] == pid]["title"].iloc[0]

    # return empty string if can't find title
    return ""


@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def t1_pred_title(input_value):

    title = get_article_title(input_value, 1)
    response = get_article_response(input_value, 1)
    claps = get_article_claps(input_value, 1)

    article = title + " (%s response, %s claps)" % (response, claps)

    # return article title with rank 1
    return article

@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def t1_pred_title_cold(input_value):

    title = get_article_title_cold(input_value, 1)
    response = get_article_response_cold(input_value, 1)
    claps = get_article_claps_cold(input_value, 1)

    article = title + " (%s response, %s claps)" % (response, claps)
    # return article title with rank 1
    return article


@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def t2_pred_title(input_value):

    title = get_article_title(input_value, 2)
    response = get_article_response(input_value, 2)
    claps = get_article_claps(input_value, 2)

    article = title + " (%s response, %s claps)" % (response, claps)

    # return article title with rank 2
    return article

@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def t2_pred_title_cold(input_value):

    title = get_article_title_cold(input_value, 2)
    response = get_article_response_cold(input_value, 2)
    claps = get_article_claps_cold(input_value, 2)

    article = title + " (%s response, %s claps)" % (response, claps)
    # return article title with rank 2
    return article

@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def t3_pred_title(input_value):

    title = get_article_title(input_value, 3)
    response = get_article_response(input_value, 3)
    claps = get_article_claps(input_value, 3)

    article = title + " (%s response, %s claps)" % (response, claps)

    # return article title with rank 3
    return article

@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def t3_pred_title_cold(input_value):

    title = get_article_title_cold(input_value, 3)
    response = get_article_response_cold(input_value, 3)
    claps = get_article_claps_cold(input_value, 3)

    article = title + " (%s response, %s claps)" % (response, claps)
    # return article title with rank 3
    return article


def get_article_claps(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map_ws[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[:3][0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map_ws.iteritems():
        if pidx ==  post_index:

            # return claps for recommended item
            return  df.loc[df["post_id"] == pid]["claps"].iloc[0]

    # return empty string if can't find claps
    return ""

def get_article_claps_cold(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map_cs[user]                                  # internal index of user
    user_prediction = np.array(prediction_cs[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[:3][0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map_cs.iteritems():
        if pidx ==  post_index:

            # return claps for recommended item
            return  df.loc[df["post_id"] == pid]["claps"].iloc[0]

    # return empty string if can't find claps
    return ""


def get_article_response(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map_ws[user]                                  # internal index of user
    user_prediction = np.array(prediction_ws[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[:3][0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map_ws.iteritems():
        if pidx ==  post_index:

            # return response for recommended item
            return  df.loc[df["post_id"] == pid]["response"].iloc[0]

    # return empty string if can't find response
    return ""

def get_article_response_cold(input_value, rank):

    # get prediction for input user
    user = input_value
    user_index = user_id_map_cs[user]                                  # internal index of user
    user_prediction = np.array(prediction_cs[user_index].todense())    # all predictions for this user
    user_top_n_index = np.argsort(-user_prediction)[:3][0]          # top recommendation
    post_index = user_top_n_index[rank]                             # index of top recommendation

    # loop over item_id_map, find post_id given index
    for pid, pidx in item_id_map_cs.iteritems():
        if pidx ==  post_index:

            # return response for recommended item
            return  df.loc[df["post_id"] == pid]["response"].iloc[0]

    # return empty string if can't find response
    return ""

if __name__ == '__main__':
    app.run_server()






#ADD DATES
#PUT DATES, RESPONSES, CLAPS ON THE NEXT LINE










