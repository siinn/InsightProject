import flask
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import os, glob, base64
import pandas as pd
from read_data import read_data
from dash_get_articles import get_article_info, get_article_info_cold
import json

#---------------------------------------------
# load global data
#---------------------------------------------

# load dataset
df_user = read_data("user_detail_medium")

# create app
application = flask.Flask(__name__)
app = dash.Dash(__name__, static_folder='static', server=application)

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

        # main content
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
                            {'label': 'Ruslan Nikolaev', 'value': 'ed47957c7bbd'},
                            {'label': 'Kerry Benjamin', 'value': '78ff5a2855cb'},
                            {'label': 'Beau Gordon', 'value': '8a5d3f4a4655'},
                            {'label': 'Tarık Uygun', 'value': '29d86acdab2b'},
                            {'label': 'MINIMAL', 'value': 'd3e12cffc59a'}
                        ],
                        value='ed47957c7bbd',
                    ),
                ], style={'display': 'block','padding': '10px 0px 10x 0px'}),

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

                #html.P(id='usergroup'),


            ], style={'max-width': '400px', 'width': '20%','display': 'inline-block', "float":"left",'padding': '0px 25px 0px 0px'}),


            # print user information
            html.Div([
                html.H2(id='username'),
            ], style={'width': '60%', 'display': 'inline-block','vertical-align': 'top','padding': '50px 0px 0px 50px'}),


            # recommendation output
            html.Div([

                html.H3('Recommendation (Popular)'),
                html.Li(html.A(id='t1_pred_link_cold', target="_blank", children="dummy", style={"text-decoration": "none"})),
                html.Li(html.A(id='t2_pred_link_cold', target="_blank", children="dummy", style={"text-decoration": "none"})),
                html.Li(html.A(id='t3_pred_link_cold', target="_blank", children="dummy", style={"text-decoration": "none"})),


            #], style={'width': '500px', 'display': 'inline-block', 'vertical-align': 'top','padding': '30px 0px 0px 50px'}),
            ], style={'width': '30%', 'display': 'inline-block', "float":"left", "margin":"0 auto",'vertical-align': 'top','padding': '30px 0px 0px 50px'}),

            html.Div([

                html.H3('Recommendation (Customized)'),
                html.Li(html.A(id='t1_pred_link', target="_blank", children="dummy", style={"text-decoration": "none"})),
                html.Li(html.A(id='t2_pred_link', target="_blank", children="dummy", style={"text-decoration": "none"})),
                html.Li(html.A(id='t3_pred_link', target="_blank", children="dummy", style={"text-decoration": "none"})),

            #], style={'width': '500px', 'display': 'inline-block', 'vertical-align': 'top','padding': '30px 0px 0px 50px'}),
            ], style={'width': '30%', 'display': 'inline-block', "float":"left", "margin":"0 auto",'vertical-align': 'top','padding': '30px 0px 0px 50px'}),

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
        ], style={'width': '100%', 'margin': '100px 0px 0px 0px', 'text-align': 'center', 'vertical-align': 'top', 'display':'block'})


    # end of wrapper
    #], style={'width': '1600px', "position":"relative", "float":"left", "margin":"0 auto", 'padding': '0px 50px 0px 50px', "box-sizing":"border-box"}),
    ], style={'max-width': '2000px', 'width': '100%', "position":"relative", "float":"left", "margin":"0 auto", 'padding': '0px 50px 0px 50px', "box-sizing":"border-box"}),


    # hidden element for storing data
    html.Div(id='t1_pred_info', style={'display': 'none'}),
    html.Div(id='t2_pred_info', style={'display': 'none'}),
    html.Div(id='t3_pred_info', style={'display': 'none'}),
    html.Div(id='t1_pred_info_cold', style={'display': 'none'}),
    html.Div(id='t2_pred_info_cold', style={'display': 'none'}),
    html.Div(id='t3_pred_info_cold', style={'display': 'none'})
])


#---------------------------------------------
# callback function for general message
#---------------------------------------------

# welcome message for user
@app.callback(
    Output(component_id='username', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def update_username(input_value):
    name = df_user.loc[df_user["user_id"]==input_value, ["name"]].iloc[0][0]
    return 'Hi, {}! Here are some interesting articles to read.'.format(name)

# set page number
@app.callback(
    Output('page_number', 'children'), 
    [Input('next_page', 'n_clicks'), Input('previous_page', 'n_clicks')])
def set_page_number(n_next_page, n_prev_page):
    # calculate page number
    page = int(n_next_page - n_prev_page + 1)
    return page if page > 1 else 1



#---------------------------------------------
# callback function for combined article info
#---------------------------------------------

# recommendation info 1 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_info', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children'),
     Input(component_id='topic', component_property='value')]
)
def t1_pred_info(input_value, page_number, topic):

    # rank to return
    rank = 1 + 3 * (page_number-1)
    # return article info with rank 1
    return get_article_info(input_value, rank, topic)

# recommendation info 1 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_info_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children'),
     Input(component_id='topic', component_property='value')]
)
def t1_pred_info_cold(input_value, page_number, topic):
    # rank to return
    rank = 1 + 3 * (page_number-1)
    # return article info with rank 1
    return get_article_info_cold(input_value, rank, topic)

# recommendation info 2 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_info', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children'),
     Input(component_id='topic', component_property='value')]
)
def t2_pred_info(input_value, page_number, topic):
    # rank to return
    rank = 2 + 3 * (page_number-1)
    # return article info with rank 2
    return get_article_info(input_value, rank, topic)

# recommendation info 2 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_info_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children'),
     Input(component_id='topic', component_property='value')]
)
def t2_pred_info_cold(input_value, page_number, topic):
    # rank to return
    rank = 2 + 3 * (page_number-1)
    # return article info with rank 2
    return get_article_info_cold(input_value, rank, topic)

# recommendation info 3 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_info', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children'),
     Input(component_id='topic', component_property='value')]
)
def t3_pred_info(input_value, page_number, topic):
    # rank to return
    rank = 3 + 3 * (page_number-1)
    # return article info with rank 3
    return get_article_info(input_value, rank, topic)

# recommendation info 3 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_info_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children'),
     Input(component_id='topic', component_property='value')]
)
def t3_pred_info_cold(input_value, page_number, topic):
    # rank to return
    rank = 3 + 3 * (page_number-1)
    # return article info with rank 3
    return get_article_info_cold(input_value, rank, topic)







#---------------------------------------------
# callback function for recommendations (link)
#---------------------------------------------

# recommendation link 1 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='href'),
    [Input(component_id='t1_pred_info', component_property='children')]
)
def t1_pred_link(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article link with rank 1
    return article_info["link"]

# recommendation link 1 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link_cold', component_property='href'),
    [Input(component_id='t1_pred_info_cold', component_property='children')]
)
def t1_pred_link_cold(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article link with rank 1
    return article_info["link"]


# recommendation link 2 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='href'),
    [Input(component_id='t2_pred_info', component_property='children')]
)
def t2_pred_link(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article link with rank 2
    return article_info["link"]

# recommendation link 2 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link_cold', component_property='href'),
    [Input(component_id='t2_pred_info_cold', component_property='children')]
)
def t2_pred_link_cold(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article link with rank 2
    return article_info["link"]


# recommendation link 3 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='href'),
    [Input(component_id='t3_pred_info', component_property='children')]
)
def t3_pred_link(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article link with rank 3
    return article_info["link"]

# recommendation link 3 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link_cold', component_property='href'),
    [Input(component_id='t3_pred_info_cold', component_property='children')]
)
def t3_pred_link_cold(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article link with rank 3
    return article_info["link"]

#---------------------------------------------
# callback function for recommendations (title)
#---------------------------------------------

# recommendation title 1 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='children'),
    [Input(component_id='t1_pred_info', component_property='children')]
)
def t1_pred_title(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article title with rank 1
    return article_info["title"] + " (%s response, %s claps)" % (article_info["response"], article_info["claps"])

# recommendation title 1 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link_cold', component_property='children'),
    [Input(component_id='t1_pred_info_cold', component_property='children')]
)
def t1_pred_title_cold(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article title with rank 1
    return article_info["title"] + " (%s response, %s claps)" % (article_info["response"], article_info["claps"])

# recommendation title 2 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='children'),
    [Input(component_id='t2_pred_info', component_property='children')]
)
def t2_pred_title(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article title with rank 2
    return article_info["title"] + " (%s response, %s claps)" % (article_info["response"], article_info["claps"])

# recommendation title 2 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link_cold', component_property='children'),
    [Input(component_id='t2_pred_info_cold', component_property='children')]
)
def t2_pred_title_cold(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article title with rank 2
    return article_info["title"] + " (%s response, %s claps)" % (article_info["response"], article_info["claps"])



# recommendation title 3 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='children'),
    [Input(component_id='t3_pred_info', component_property='children')]
)
def t3_pred_title(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article title with rank 3
    return article_info["title"] + " (%s response, %s claps)" % (article_info["response"], article_info["claps"])

# recommendation title 3 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link_cold', component_property='children'),
    [Input(component_id='t3_pred_info_cold', component_property='children')]
)
def t3_pred_title_cold(article_info):

    # convert stored json to dict
    article_info = json.loads(article_info)

    # return article title with rank 3
    return article_info["title"] + " (%s response, %s claps)" % (article_info["response"], article_info["claps"])





# run main app
if __name__ == '__main__':

    # local testing
    #app.run_server()
    application.run(host='0.0.0.0')





#ADD DATES










