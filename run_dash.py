import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import os, glob, base64
import pandas as pd
from read_data import read_data
from dash_get_articles import get_article_link, get_article_link_cold
from dash_get_articles import get_article_title, get_article_title_cold
from dash_get_articles import get_article_claps, get_article_claps_cold 
from dash_get_articles import get_article_response, get_article_response_cold

#---------------------------------------------
# load global data
#---------------------------------------------

# load dataset
df_user = read_data("user_detail_medium")

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

                #html.P(id='usergroup'),


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

            ], style={'width': '500px', 'display': 'inline-block', 'vertical-align': 'top','padding': '30px 0px 0px 50px'}),

        # end of main content
        ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),

        # footer with page button
        html.Div([

            # horizontal bar
            html.Hr(style={'padding': '0px 0px 0px 0px','margin': '0px 50px 20px 150px'}),

            html.Button('Previous', id='previous_page', n_clicks=0, style={'display': 'inline', 'margin': '0px 50px 0px 50px', 'width':'150px', 'text-align':'center'}),
            html.P("Page ", style={'display': 'inline'}),
            html.P(children=1, id='page_number', style={'display': 'inline'}),
            html.Button('Next', id='next_page', n_clicks=0, style={'display': 'inline', 'margin': '0px 50px 0px 50px', 'width':'150px', 'text-align':'center'}),
        ], style={'width': '100%', 'margin': '100px 0px 0px 150px', 'text-align': 'center', 'vertical-align': 'top', 'display':'block'})


    # end of wrapper
    ], style={"max-width":"1600px",'width': '90%', "position":"relative", "float":"left", "margin":"0 auto", 'padding': '0px 50px 0px 50px', "box-sizing":"border-box"}),


    # hidden element for storing data
    #html.Div(id='page_number', style={'display': 'none'})

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

# display user group
#@app.callback(
#    Output(component_id='usergroup', component_property='children'),
#    [Input(component_id='selected_user', component_property='value')]
#)
#def update_usergroup(input_value):
#    name = df_user.loc[df_user["user_id"]==input_value, ["group_medium"]].iloc[0][0]
#    return 'User group: "{}"'.format(name)

# set page number
@app.callback(
    Output('page_number', 'children'), 
    [Input('next_page', 'n_clicks'), Input('previous_page', 'n_clicks')])
def set_page_number(n_next_page, n_prev_page):
    # calculate page number
    page = int(n_next_page - n_prev_page + 1)
    return page if page > 1 else 1

## reset page number
#@app.callback(
#    Output('previous_page', 'n_clicks'),
#    [Input('page_number', 'children')],
#    [State('previous_page', 'n_clicks')]) 
#def reset_previous_page_click(page, n_clicks):
#    return 0 if page < 1 else n_clicks

#---------------------------------------------
# callback function for recommendations (link)
#---------------------------------------------

# recommendation link 1 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='href'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t1_pred_link(input_value, page_number):

    # rank to return
    rank = 1 + 3 * (page_number-1)
    # return article link with rank 1
    return get_article_link(input_value, rank)

# recommendation link 1 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link_cold', component_property='href'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t1_pred_link_cold(input_value, page_number):
    # rank to return
    rank = 1 + 3 * (page_number-1)
    # return article link with rank 1
    return get_article_link_cold(input_value, rank)

# recommendation link 2 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='href'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t2_pred_link(input_value, page_number):
    # rank to return
    rank = 2 + 3 * (page_number-1)
    # return article link with rank 2
    return get_article_link(input_value, rank)

# recommendation link 2 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link_cold', component_property='href'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t2_pred_link_cold(input_value, page_number):
    # rank to return
    rank = 2 + 3 * (page_number-1)
    # return article link with rank 2
    return get_article_link_cold(input_value, rank)

# recommendation link 3 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='href'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t3_pred_link(input_value, page_number):
    # rank to return
    rank = 3 + 3 * (page_number-1)
    # return article link with rank 3
    return get_article_link(input_value, rank)

# recommendation link 3 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link_cold', component_property='href'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t3_pred_link_cold(input_value, page_number):
    # rank to return
    rank = 3 + 3 * (page_number-1)
    # return article link with rank 3
    return get_article_link_cold(input_value, rank)

#---------------------------------------------
# callback function for recommendations (title)
#---------------------------------------------

# recommendation title 1 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t1_pred_title(input_value, page_number):

    # rank to return
    rank = 1 + 3 * (page_number-1)

    title = get_article_title(input_value, rank)
    response = get_article_response(input_value, rank)
    claps = get_article_claps(input_value, rank)

    article = title + " (%s response, %s claps)" % (response, claps)

    # return article title with rank 1
    return article

# recommendation title 1 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t1_pred_link_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t1_pred_title_cold(input_value, page_number):

    # rank to return
    rank = 1 + 3 * (page_number-1)

    title = get_article_title_cold(input_value, rank)
    response = get_article_response_cold(input_value, rank)
    claps = get_article_claps_cold(input_value, rank)

    article = title + " (%s response, %s claps)" % (response, claps)
    # return article title with rank 1
    return article


# recommendation title 2 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t2_pred_title(input_value, page_number):

    # rank to return
    rank = 2 + 3 * (page_number-1)

    title = get_article_title(input_value, rank)
    response = get_article_response(input_value, rank)
    claps = get_article_claps(input_value, rank)

    article = title + " (%s response, %s claps)" % (response, claps)

    # return article title with rank 2
    return article

# recommendation title 2 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t2_pred_link_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t2_pred_title_cold(input_value, page_number):

    # rank to return
    rank = 2 + 3 * (page_number-1)

    title = get_article_title_cold(input_value, rank)
    response = get_article_response_cold(input_value, rank)
    claps = get_article_claps_cold(input_value, rank)

    article = title + " (%s response, %s claps)" % (response, claps)
    # return article title with rank 2
    return article


# recommendation title 3 (improved)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t3_pred_title(input_value, page_number):

    # rank to return
    rank = 3 + 3 * (page_number-1)

    title = get_article_title(input_value, rank)
    response = get_article_response(input_value, rank)
    claps = get_article_claps(input_value, rank)

    article = title + " (%s response, %s claps)" % (response, claps)

    # return article title with rank 3
    return article

# recommendation title 3 (cold start)
@app.callback(
    dash.dependencies.Output(component_id='t3_pred_link_cold', component_property='children'),
    [Input(component_id='selected_user', component_property='value'),
     Input(component_id='page_number', component_property='children')]
)
def t3_pred_title_cold(input_value, page_number):

    # rank to return
    rank = 3 + 3 * (page_number-1)

    title = get_article_title_cold(input_value, rank)
    response = get_article_response_cold(input_value, rank)
    claps = get_article_claps_cold(input_value, rank)

    article = title + " (%s response, %s claps)" % (response, claps)
    # return article title with rank 3
    return article






# run main app
if __name__ == '__main__':
    app.run_server()






#ADD DATES










