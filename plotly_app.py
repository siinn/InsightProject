import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pickle
import os, glob
import pandas as pd
import base64
from read_data import read_data


# load dataset
df = read_data("list_articles")
df_user = read_data("user_detail_medium")


# create app
app = dash.Dash(__name__, static_folder='static')

# options to use local css
app.scripts.config.serve_locally=True
app.css.config.serve_locally=True
#app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

# custoom local image
image_filename = 'static/analytics_g.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# create layout
app.layout = html.Div([
    
    # load custom css
    html.Link(href='static/main.css', rel='stylesheet'),

    # header
    html.Div([
        html.H2('Recommendation system'),
        html.H3('Some description',)
    ], style={'width': '90%', 'padding': '0px 50px 0px 50px'}),

    html.Hr(style={'padding': '0px 0px 0px 0px'}),

    # container
    html.Div([

        # user selection
        html.Div([

            # logo
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width="100px", style={'display':'block'}),
                html.H1('Medium', style={'display':'inline'}),
            ], style={'width': '300px','display': 'inline-block'}),

            # drop dropdown and multi-select dropdown
            html.Div([
                html.Label('Dropdown'),
                dcc.Dropdown(
                    id='selected_user',
                    options=[
                        {'label': 'User1', 'value': '54a9d67e5f79'},
                        {'label': 'User2', 'value': '8a5d3f4a4655'},
                        {'label': 'User3', 'value': '29d86acdab2b'}
                    ],
                    value='54a9d67e5f79',
                ),
            ], style={'display': 'block','padding': '10px 0px 10x 0px'}),

            html.Div([
                html.Label('Multi-Select Dropdown'),
                dcc.Dropdown(
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    value=['MTL', 'SF'],
                    multi=True
                ),
            ], style={'display': 'block','padding': '0px 0px 10px 0px'}),

            # custom user name and slider bar
            html.Div([

                html.Label('Text Input'),
                dcc.Input(id='selected_user', value='29d86acdab2b', type='text'),
            ], style={'display': 'block','padding': '0px 0px 10px 0px'}),

            html.Div([

                html.Label('Slider'),
                dcc.Slider(
                    min=0,
                    max=2,
                    marks={0:"Novice", 1:"Data related", 2:"Data scientist"},
                    value=0,
                ),
            ], style={'display': 'block','padding': '0px 0px 10px 0px'}),

        ], style={'width': '300px','display': 'inline-block', "float":"left",'padding': '0px 25px 0px 0px'}),

        # recommendation output
        html.Div([

            html.H3('Recommendation'),
            html.H4(id='print_selected_user'),
            html.H4(id='username'),
            html.H4(id='usergroup'),
            html.H4(id='n_pred1'),
            html.A("Link to external site", href='https://plot.ly', target="_blank")

        ], style={'width': '400px', 'display': 'inline-block', 'vertical-align': 'top','padding': '100px 0px 0px 50px'}),

        html.Div([

            html.H3('Improved recommendation'),
            html.H4(id='print_selected_user'),
            html.H4(id='username'),
            html.H4(id='usergroup'),
            html.P(id='n_pred1')

        ], style={'width': '400px', 'display': 'inline-block', 'vertical-align': 'top','padding': '100px 0px 0px 50px'})

    ], style={"max-width":"1600px",'width': '90%', "position":"relative", "float":"left", "margin":"0 auto", 'padding': '0px 50px 0px 50px', "box-sizing":"border-box"}),
    #], className="container"),
])



@app.callback(
    Output(component_id='print_selected_user', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def update_username(input_value):
    return 'User ID: "{}"'.format(input_value)


@app.callback(
    Output(component_id='username', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def update_username(input_value):
    name = df_user.loc[df_user["user_id"]==input_value, ["name"]].iloc[0][0]
    return 'User name: "{}"'.format(name)




@app.callback(
    Output(component_id='usergroup', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def update_usergroup(input_value):
    name = df_user.loc[df_user["user_id"]==input_value, ["group_medium"]].iloc[0][0]
    return 'User group: "{}"'.format(name)



@app.callback(
    Output(component_id='n_pred1', component_property='children'),
    [Input(component_id='selected_user', component_property='value')]
)
def n_prediction(input_value):

    # retrieve pickled model
    p = open('data/pickle/top_n','rb')
    top_n = pickle.load(p)

    # top recommendation
    pred1 = top_n[input_value][0][0]

    # fine link
    link1 = df.loc[df["post_id"]==pred1, ["link"]].iloc[0][0]

    # return recommendation for this user
    return 'Recommendation 1: "{}"'.format(link1)




if __name__ == '__main__':
    app.run_server()




