import flask
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import os, glob, base64
import pandas as pd
from read_data import read_data
import json
from datetime import datetime as dt
from datetime import date
from datetime import timedelta
import bz2
import pickle
from dateutil import parser
from collections import defaultdict
import pytz

#---------------------------------------------
# load global data
#---------------------------------------------

# local path for data
local_data = "/home/ubuntu/InsightProject/"
#local_data = "/Users/sche/Documents/data_science/insight/medium_recommendation/"

# load dataset
df_user = read_data("user_detail_medium")
df_articles= read_data("list_articles")

# remove duplicates
df_articles = df_articles.drop_duplicates(subset=["post_id"], keep="first")

# convert string to datetime
df_articles["post_time"] = df_articles["post_time"].apply(parser.parse)

# load pickled user names
p = bz2.BZ2File(local_data+'data/pickle/user_feature_names','rb')
user_feature_names = pickle.load(p)
p.close()

# load pickled prediction for hypothetical user
p = bz2.BZ2File(local_data+'data/pickle/prediction_hypo','rb')
prediction_hypo = pickle.load(p, encoding='latin1')
p.close()

# get last update date
with open ("log_model_update", "rb") as f:
	last_update = str(f.readlines()[-1]).strip()
last_update = last_update.strip("b'\\n'")
last_update = "Model built on " + last_update

# set number of articles to display per page
max_article = 6
page_correction = 1

# create app
application = flask.Flask(__name__)
app = dash.Dash(__name__, static_folder='static', server=application)

# goverment/health theme
app.css.append_css({'external_url': 'https://codepen.io/plotly/pen/EQZeaW.css'})
app.title='Warm Welcome to Medium'

# custoom local image
image_filename = 'static/new-york-city-H.jpeg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# create layout
app.layout = html.Div([

    # container
    html.Div([

        # header image
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width="100%", style={'display':'block'}),
        ], style={'width':'100%','display': 'inline-block','padding': '0px 0px 0px 0px', 'vertical-align': 'top'}),

        # main content
        html.Div([


            # horizontal bar 
            html.Hr(style={'padding': '0px 0px 0px 0px','margin': '0px 0px 0px 0px'}),

            # top panel
            html.Div([

                # leftside of top panel, logo
                html.Div([
                    #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width="100px", style={'display':'block'}),
                    html.H1('Medium', style={'margin': '0px 0px 0px 0px'}),
                    html.H6('Improved recommender system'),

				    # display last update time
                    html.H6(last_update)
                ], style={'width':'30%','display': 'inline-block','padding': '0px 0px 0px 0px', 'vertical-align': 'top'}),
		

				# rightside of top panel
                html.Div([
                	html.Div([
                	    html.Label('What\'s your interest?'),
                	    dcc.Dropdown(
                	        id='topic',
                	        options=[
                	            {"label": "deep-learning", "value": "deep-learning"},
                	            {"label": "artificial-intelligence", "value": "artificial-intelligence"},
                	            {"label": "machine-learning", "value": "machine-learning"},
                	            {"label": "data-science", "value": "data-science"},
                	            {"label": "neural-networks", "value": "neural-networks"},
                	        ],
                	        value=['data-science', 'machine-learning'],
                	        clearable=False,
                	        multi=True
                	    ),
                	], style={'width':'45%', 'display': 'inline-block','padding': '10px 10px 10px 10px', 'vertical-align': 'top'}),

                	# date picker
			    	html.Div([
                	    html.Label('Choose dates'),
			    	    dcc.DatePickerRange(
			    	        id='my-date-picker-range',
			    	        min_date_allowed=dt(2012, 1, 1),
			    	        max_date_allowed=date.today(),
			    	        initial_visible_month=date.today(),
			    	        start_date= date.today() - timedelta(days=120),
			    	        end_date= date.today(),
			    	    ),
			    	], style={'width':'45%', 'display': 'inline-block','vertical-align': 'top','padding': '10px 10px 10px 10px', 'vertical-align': 'top'}),

					# detailed input box	
                	html.Div([
						html.Details([
							html.Summary('Advanced options'),

						# user feature selection
                		html.Div([
                		    html.Label('What describes you the best?'),
                	        dcc.Dropdown(
                		        id='user_feature',
							    value=['data_science', 'machine_learning'],
							    #labelStyle={'display': 'inline-block', 'padding': '0px 10px 0px 10px',}
                	            multi=True
							)
                		], style={'width':'45%','display': 'none','padding': '10px 10px 10px 10px', 'vertical-align': 'top'}),

						])
                	], style={'width':'79%','display': 'none','padding': '10px 0px 0px 10px', 'vertical-align': 'top'}),

				# end of rightside panel
                ], style={'width':'70%','display': 'inline-block','padding': '0px 0px 0px 0px', 'vertical-align': 'top'}),

			# end of top panel
            ], style={'width': '100%','display': 'inline-block', "float":"left",'padding': '30px 0px 0px 0px', 'vertical-align': 'top'}),


            # print user information
            html.Div([
                html.H2("Hi! Here are some interesting articles to read."),
            ], style={'width': '100%', 'display': 'inline-block','vertical-align': 'top',  'padding': '30px 0px 30px 0px'}),


            # recommendation output in two columns
            html.Div([

                html.Li(html.A(id='t0_pred_link', target="_blank", children="Loading..", style={"text-decoration": "none"})),
                html.Li(html.A(id='t1_pred_link', target="_blank", children="Loading..", style={"text-decoration": "none"})),
                html.Li(html.A(id='t2_pred_link', target="_blank", children="Loading..", style={"text-decoration": "none"})),

            ], style={'width': '40%', 'display': 'inline-block', "float":"left", "margin":"0 auto",'vertical-align': 'top','margin': '30px 50px 50px 0px'}),

            html.Div([
                html.Li(html.A(id='t3_pred_link', target="_blank", children="Loading..", style={"text-decoration": "none"})),
                html.Li(html.A(id='t4_pred_link', target="_blank", children="Loading..", style={"text-decoration": "none"})),
                html.Li(html.A(id='t5_pred_link', target="_blank", children="Loading..", style={"text-decoration": "none"})),

            ], style={'width': '40%', 'display': 'inline-block', "float":"left", "margin":"0 auto",'vertical-align': 'top','margin': '30px 0px 50px 50px'}),

        # end of main content
        ], style={'width': '100%', "float":"left", "margin":"0 auto",'display': 'inline-block', 'vertical-align': 'top'}),

        # footer with page button
        html.Div([

            # horizontal bar
            html.Hr(style={'padding': '0px 0px 0px 0px','margin': '0px 50px 20px 150px'}),

            html.Button('Previous', id='previous_page', n_clicks=0, style={'display': 'inline', 'margin': '0px 50px 0px 50px', 'width':'150px', 'text-align':'center'}),
            html.P("Page ", style={'display': 'inline'}),
            html.P(children=1, id='page_number', style={'display': 'inline'}),
            html.Button('Next', id='next_page', n_clicks=0, style={'display': 'inline', 'margin': '0px 50px 0px 50px', 'width':'150px', 'text-align':'center'}),
        ], style={'width': '100%', 'margin': '100px 0px 0px 0px', 'text-align': 'center', 'vertical-align': 'top', 'display':'block'}),


    # hidden element for storing data
    html.Div(id='t1_pred_info', style={'display': 'none'}),
    html.Div(id='t2_pred_info', style={'display': 'none'}),
    html.Div(id='t3_pred_info', style={'display': 'none'}),
    html.H1(id='user_hypo', style={'display': 'none'}),
    html.H1(id='user_df', style={'display': 'none'}),

    # end of wrapper
    ], style={'max-width': '1300px', 'width': '80%', "float":"center", "margin":"0 auto", 'padding': '0px 0px 0px 0px', "box-sizing":"border-box", 'vertical-align': 'top', }),

# end of HTML
], style={'width': '100%', 'vertical-align': 'top', 'padding': '0px 0px 0px 0px'})


#---------------------------------------------
# callback function for general message
#---------------------------------------------

# set page number
@app.callback(
    Output('page_number', 'children'), 
    [Input('next_page', 'n_clicks'), Input('previous_page', 'n_clicks')])
def set_page_number(n_next_page, n_prev_page):
    # calculate page number
    page = int(n_next_page - n_prev_page + 1)
    return page if page > 1 else 1


# set user based on user description and feature dropdown menu
@app.callback(
    Output(component_id='user_hypo', component_property='children'),
    [Input(component_id='user_feature', component_property='value'),]
)
def set_user(user_feature):

    users_hypo = []
    
    # use feature dropdown menu to select hypothetical user
    for index_feature, feature in enumerate(user_feature_names):    # loop over user features from pickle
        if (feature in user_feature):                              
            users_hypo.append(str(index_feature))
    
    return users_hypo

#---------------------------------------------
# callback function for combined article info
#---------------------------------------------
# given users, find best predictions
def find_best_predictions(users):

    prediction_from_all_users = defaultdict(list)
    prediction_best = {}
   
    for user in users:

        # prediction for this user
        dict_pred_user = dict(prediction_hypo[int(user)])

        # swap key and value
        dict_pred_user = dict(zip(dict_pred_user.values(),dict_pred_user.keys()))

        # merge predictions into one dictionary
        for key, value in dict_pred_user.items():
            prediction_from_all_users[key].append(value)

    # find the maximum score from each user
    for post, scores in prediction_from_all_users.items():
        prediction_best[post] = max(scores)

    return prediction_best


# create dataframe that contains predictions for given user
@app.callback(
	Output('user_df', 'children'),
    [Input(component_id='user_hypo', component_property='children')]
	)
def create_dataframe(users):

    max_prediction = 500

    # find best prediction for this set of users
    prediction = find_best_predictions(users)

    # create new dataframe to hold predictions
    df_prediction = pd.DataFrame()

    # loop over predictions to create dataframe
    for i in range(max_prediction):
        df_prediction = pd.concat([df_prediction,df_articles.loc[df_articles["post_id"]==list(prediction)[i]]])
        
    return df_prediction.to_json(date_format='iso', orient='split')



#---------------------------------------------
# callback function for recommendations (title)
#---------------------------------------------

# recommendation title 0
@app.callback(
    dash.dependencies.Output(component_id='t0_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t0_pred_title(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 0 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction 0
    if (len(df) > rank):
        article = df.iloc[rank]

        # return article title with rank 0
        return article["title"] + ", %s (%s response, %s claps)" % (article["post_time"].strftime("%B %d, %Y"), article["response"], article["claps"])

    else:
        return "No article available that matches your selection"



# recommendation title 1
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t1_pred_title(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 1 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction 1
    if (len(df) > rank):
        article = df.iloc[rank]

        # return article title with rank 1
        return article["title"] + ", %s (%s response, %s claps)" % (article["post_time"].strftime("%B %d, %Y"), article["response"], article["claps"])

    else:
        return "No article available that matches your selection"




# recommendation title 2
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t2_pred_title(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 2 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction 2
    if (len(df) > rank):
        article = df.iloc[rank]

        # return article title with rank 2
        return article["title"] + ", %s (%s response, %s claps)" % (article["post_time"].strftime("%B %d, %Y"), article["response"], article["claps"])

    else:
        return "No article available that matches your selection"




# recommendation title 3
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t3_pred_title(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 3 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction 3
    if (len(df) > rank):
        article = df.iloc[rank]

        # return article title with rank 3
        return article["title"] + ", %s (%s response, %s claps)" % (article["post_time"].strftime("%B %d, %Y"), article["response"], article["claps"])

    else:
        return "No article available that matches your selection"




# recommendation title 4
@app.callback(
    dash.dependencies.Output(component_id='t4_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t4_pred_title(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 4 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction 4
    if (len(df) > rank):
        article = df.iloc[rank]

        # return article title with rank 4
        return article["title"] + ", %s (%s response, %s claps)" % (article["post_time"].strftime("%B %d, %Y"), article["response"], article["claps"])

    else:
        return "No article available that matches your selection"



# recommendation title 5
@app.callback(
    dash.dependencies.Output(component_id='t5_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t5_pred_title(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 5 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction 5
    if (len(df) > rank):
        article = df.iloc[rank]

        # return article title with rank 5
        return article["title"] + ", %s (%s response, %s claps)" % (article["post_time"].strftime("%B %d, %Y"), article["response"], article["claps"])

    else:
        return "No article available that matches your selection"


#---------------------------------------------
# callback function for recommendations (link)
#---------------------------------------------

# recommendation link 0 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t0_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t0_pred_link(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 0 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction
    if (len(df) > rank):
        article = df.iloc[rank]

    # retrieve prediction 0
    return article["link"]




# recommendation link 1 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t1_pred_link(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 1 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction
    if (len(df) > rank):
        article = df.iloc[rank]

    # retrieve prediction 1
    return article["link"]





# recommendation link 2 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t2_pred_link(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 2 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction
    if (len(df) > rank):
        article = df.iloc[rank]

    # retrieve prediction 2
    return article["link"]






# recommendation link 3 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t3_pred_link(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 3 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction
    if (len(df) > rank):
        article = df.iloc[rank]

    # retrieve prediction 3
    return article["link"]






# recommendation link 4 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t4_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t4_pred_link(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 4 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction
    if (len(df) > rank):
        article = df.iloc[rank]

    # retrieve prediction 4
    return article["link"]





# recommendation link 5 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t5_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children'),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input(component_id='topic', component_property='value'),
     ]
)
def t5_pred_link(user_df_json, page_number, start_date, end_date, topic):

    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    # rank
    rank = 5 + max_article * (page_number-page_correction)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # filter by date
    df = df.loc[(df["post_time"] > start_date) & (df["post_time"] < end_date)]

    # filter by topic
    df = df.loc[df["topic"].apply(lambda x: x in topic)]

    # retrieve prediction
    if (len(df) > rank):
        article = df.iloc[rank]

    # retrieve prediction 5
    return article["link"]












# run main app
if __name__ == '__main__':

    # local testing
    #app.run_server()
    application.run(host='0.0.0.0')





#ADD DATES










