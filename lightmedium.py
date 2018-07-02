import flask
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import os, glob, base64
import pandas as pd
from read_data import read_data
from dash_get_articles import get_article_info
import json
from datetime import datetime as dt
import pickle
from collections import defaultdict

#---------------------------------------------
# load global data
#---------------------------------------------

# load dataset
df_user = read_data("user_detail_medium")
df_articles= read_data("list_articles")

# remove duplicates
df_articles = df_articles.drop_duplicates(subset=["post_id"], keep="first")

# load pickled user names
p = open('data/pickle/user_feature_names','rb')
user_feature_names = pickle.load(p)
p.close()

# load pickled prediction for hypothetical user
p = open('data/pickle/prediction_hypo','rb')
prediction_hypo = pickle.load(p, encoding='latin1')
p.close()

# get last update date
with open ("log_model_update", "rb") as f:
	last_update = str(f.readlines()[-1]).strip()
last_update = last_update.strip("b'\\n'")
last_update = "Model built on " + last_update

# set number of articles to display per page
max_article = 6

# create app
application = flask.Flask(__name__)
app = dash.Dash(__name__, static_folder='static', server=application)

# goverment/health theme
app.css.append_css({'external_url': 'https://codepen.io/plotly/pen/EQZeaW.css'})

# custoom local image
image_filename = 'static/new-york-city-H.jpeg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# create layout
app.layout = html.Div([

    # container
    html.Div([

        # header image
        html.Div([
            #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width="100%", style={'display':'block'}),
            #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width="100%", style={'display':'block'}),
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
		

                # user names: to be deleted
                html.Div([
                    html.Label('Select user'),
                    dcc.Dropdown(
                        id='selected_user',
                        options=[
                            {'label': 'Ruslan Nikolaev', 'value': 'ed47957c7bbd'},
                            {'label': 'Kerry Benjamin', 'value': '78ff5a2855cb'},
                            {'label': 'Beau Gordon', 'value': '8a5d3f4a4655'},
                            {'label': 'Tarık Uygun', 'value': '29d86acdab2b'},
                            {'label': 'MINIMAL', 'value': 'd3e12cffc59a'}
                        ],
                        value='ed47957c7bbd',
                    ),
                #], style={'width':'200px', 'display': 'inline-block','padding': '10px 0px 10x 0px', 'vertical-align': 'top'}),
                ], style={'width':'200px', 'display': 'none','padding': '10px 0px 10x 0px', 'vertical-align': 'top'}),

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
                	        value=['data-science', 'machine-learning', 'deep-learning'],
                	        clearable=False,
                	        multi=True
                	    ),
                	], style={'width':'45%', 'display': 'inline-block','padding': '10px 10px 10px 10px', 'vertical-align': 'top'}),

                	# date picker
			    	html.Div([
                	    html.Label('Choose dates'),
			    	    dcc.DatePickerRange(
			    	        id='my-date-picker-range',
			    	        min_date_allowed=dt(1995, 8, 5),
			    	        max_date_allowed=dt(2017, 9, 19),
			    	        initial_visible_month=dt(2017, 8, 5),
			    	        end_date=dt(2017, 8, 25)
			    	    ),
			    	], style={'width':'45%', 'display': 'inline-block','vertical-align': 'top','padding': '10px 10px 10px 10px', 'vertical-align': 'top'}),

					# detailed input box	
                	html.Div([
						html.Details([
							html.Summary('Advanced options'),

                		# user name input box
                		html.Div([
                		    html.Label('Tell me about yourself'),
							dcc.Textarea(
                		        id='user_description',
							    placeholder='Please describe yourself. "I am a happy data scientist in New York. I love to wrangle with data all day!"',
							    value='',
							    style={'width': '100%'}
							)
                		], style={'width':'45%','display': 'inline-block','padding': '10px 10px 10px 10px', 'vertical-align': 'top'}),

						# user feature selection
                		html.Div([
                		    html.Label('What describes you the best?'),
                	        dcc.Dropdown(
                		        id='user_feature',
							    value=['data_science', 'machine_learning'],
							    #labelStyle={'display': 'inline-block', 'padding': '0px 10px 0px 10px',}
                	            multi=True
							)
                		], style={'width':'45%','display': 'inline-block','padding': '10px 10px 10px 10px', 'vertical-align': 'top'}),

						])
                	], style={'width':'79%','display': 'inline-block','padding': '10px 0px 0px 10px', 'vertical-align': 'top'}),

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
    [Input(component_id='user_description', component_property='value'),
    Input(component_id='user_feature', component_property='value'),]
)
def set_user(user_description, user_feature):

    users_hypo = []
    
    # if there is no user description, use feature dropdown menu to select hypothetical user
    if (user_description == ""):
        for index_feature, feature in enumerate(user_feature_names):    # loop over user features from pickle
            if (feature in user_feature):                              
                users_hypo.append(str(index_feature))
    
    # return list of hypothetical users this user matched to	
    else:	
        for index_feature, feature in enumerate(user_feature_names):    # loop over user features from pickle
        	if (feature.replace("_", " ") in user_description):
        		users_hypo.append(str(index_feature))
    
    return users_hypo


# fill check list from user features
@app.callback(
	Output('user_feature', 'options'),
	[Input(component_id='user_description', component_property='value')]
	)
def fillChecklist(input):
    return [{'label': x.replace("_", " "), 'value' : x} for x in user_feature_names]


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

# recommendation title 0 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t0_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t0_pred_title(user_df_json, page_number):

    # rank
    rank = 0 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # retrieve prediction 0
    article = df.iloc[rank]

    # return article title with rank 0
    return article["title"] + " (%s response, %s claps)" % (article["response"], article["claps"])


# recommendation title 1 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t1_pred_title(user_df_json, page_number):

    # rank
    rank = 1 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')
    
    # retrieve prediction 1
    article = df.iloc[rank]

    # return article title with rank 1
    return article["title"] + " (%s response, %s claps)" % (article["response"], article["claps"])




# recommendation title 2 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t2_pred_title(user_df_json, page_number):

    # rank
    rank = 2 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')
    
    # retrieve prediction 2
    article = df.iloc[rank]

    # return article title with rank 2
    return article["title"] + " (%s response, %s claps)" % (article["response"], article["claps"])


# recommendation title 3 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t3_pred_title(user_df_json, page_number):

    # rank
    rank = 3 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')
    
    # retrieve prediction 3
    article = df.iloc[rank]

    # return article title with rank 3
    return article["title"] + " (%s response, %s claps)" % (article["response"], article["claps"])

# recommendation title 4 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t4_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t4_pred_title(user_df_json, page_number):

    # rank
    rank = 4 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')
    
    # retrieve prediction 4
    article = df.iloc[rank]

    # return article title with rank 4
    return article["title"] + " (%s response, %s claps)" % (article["response"], article["claps"])

# recommendation title 5 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t5_pred_link', component_property='children'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t5_pred_title(user_df_json, page_number):

    # rank
    rank = 5 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')
    
    # retrieve prediction 5
    article = df.iloc[rank]

    # return article title with rank 5
    return article["title"] + " (%s response, %s claps)" % (article["response"], article["claps"])

#---------------------------------------------
# callback function for recommendations (link)
#---------------------------------------------

# recommendation link 0 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t0_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t0_pred_link(user_df_json, page_number):

    # rank
    rank = 0 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # retrieve prediction 0
    article = df.iloc[rank]

    # return article title with rank 0
    return article["link"]


# recommendation link 1 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t1_pred_link(user_df_json, page_number):

    # rank
    rank = 1 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # retrieve prediction 1
    article = df.iloc[rank]

    # return article title with rank 1
    return article["link"]


# recommendation link 2 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t2_pred_link(user_df_json, page_number):

    # rank
    rank = 2 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # retrieve prediction 2
    article = df.iloc[rank]

    # return article title with rank 2
    return article["link"]


# recommendation link 3 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t3_pred_link(user_df_json, page_number):

    # rank
    rank = 3 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # retrieve prediction 3
    article = df.iloc[rank]

    # return article title with rank 3
    return article["link"]


# recommendation link 4 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t4_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t4_pred_link(user_df_json, page_number):

    # rank
    rank = 4 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # retrieve prediction 4
    article = df.iloc[rank]

    # return article title with rank 4
    return article["link"]


# recommendation link 5 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t5_pred_link', component_property='href'),
    [Input(component_id='user_df', component_property='children'),
     Input(component_id='page_number', component_property='children')]
)
def t5_pred_link(user_df_json, page_number):

    # rank
    rank = 5 + max_article * (page_number-1)

    # convert stored json to dataframe
    df = pd.read_json(user_df_json, orient='split')

    # retrieve prediction 5
    article = df.iloc[rank]

    # return article title with rank 5
    return article["link"]









# run main app
if __name__ == '__main__':

    # local testing
    #app.run_server()
    application.run(host='0.0.0.0')





#ADD DATES










