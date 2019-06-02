#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:27:28 2019

@author: hanxinzhang
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from textwrap import wrap
import seaborn as sns

palette = ['#7e1e9c', '#15b01a', '#ff81c0', '#653700', '#75bbfd',
           '#929591', '#029386', '#f97306', '#96f97b', '#c20078',
           '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#13eac9',
           '#ff796c', '#e6daa6', '#ceb301', '#cf6275']

sys2largeSys = {
        
        'Bacterial infections, and other intestinal infectious diseases, and stds': 'Infectious',
        'Viral infections': 'Infectious',
        'Infections caused by fungi, protozoans, worms, and infestations': 'Infectious',
        'Sequelae, and diseases classified elsewhere': 'Infectious',
        
        'Endocrine diseases': 'Endocrine, nutritional, and metabolic',
        'Nutritional diseases': 'Endocrine, nutritional, and metabolic',
        'Metabolic diseases': 'Endocrine, nutritional, and metabolic',
        
        'Diseases of the genitourinary system: pelvis, genitals and breasts': 'Genitourinary',
        'Diseases of the genitourinary system: urinary system': 'Genitourinary',
        
        'Symptoms and signs': 'Symptoms and signs',
        'Abnormal clinical and laboratory findings, not elsewhere classified': 'Symptoms and signs',
        'Ill-defined and unknown causes of mortality': 'Symptoms and signs',
        
        'External causes of morbidity and mortality': 'External causes',
        
        'Injury': 'Injury and poisoning',
        'Poisoning and certain other consequences of external causes': 'Injury and poisoning',
        
        'Certain conditions originating in the perinatal period': 'Childbirth-related',
        'Pregnancy, childbirth and the puerperium': 'Childbirth-related',
        
        'Mental and behavioural disorders': 'Mental and behavioural',
        
        'Chromosomal abnormalities, not elsewhere classified': 'Chromosomal and congenital',
        'Congenital malformations and deformations': 'Chromosomal and congenital',
        
        'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism': 'Blood and immune',
        
        'Diseases of the circulatory system': 'Circulatory',
        
        'Diseases of the digestive system': 'Digestive',
        
        'Diseases of the ear and mastoid process': 'Ear and eye',
        'Diseases of the eye and adnexa': 'Ear and eye',
        
        'Diseases of the musculoskeletal system and connective tissue': 'Musculoskeletal and connective',
        'Diseases of the nervous system': 'Nervous',
        
        'Diseases of the respiratory system': 'Respiratory',
        'Diseases of the skin and subcutaneous tissue': 'Skin and subcutaneous',
        }

# Constants preparation -------------------------------------------------------

NUM_BASES = 5
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

with open('condPeriodicsStatsData.bpkl3', 'rb') as f:
    condPeriodicsStatsData = pickle.load(f)
    
SAMPLE_SIZE = len(condPeriodicsStatsData)

np.random.seed(2019)

monthWeeks = list(np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) / 7.)
monthsPartitions = np.cumsum([0.] + monthWeeks)
monthTicks = [(x + monthsPartitions[i+1]) / 2 for i, x in enumerate(monthsPartitions[:-1])]

# Constants preparation -------------------------------------------------------
    
fourierCompX = []
condName = []
sysName = []
age = []
seasonalityPlotData = {}
condInd = []

i = 0
for k, v in condPeriodicsStatsData.items():
    
    thisCond = k.split('; ')[0]
    meanDR = v['mean DR']
    
    if (thisCond != 'All cond baseline') and (meanDR > 1e-5) :
    
        fourierCompX.append(v['Relative PQ mean'])
        condName.append(k)
        sysName.append(k.split('; ')[1].capitalize())
        age.append(k.split('; ')[-1].split('@')[-1])
        seasonalityPlotData[i] = {'mean': v['Seasonal wave mean'],
                                  'hpd': v['Seasonal wave HPD']}
        condInd.append(i)
        
        i += 1

fourierCompX = np.vstack(fourierCompX)

condInd = np.array(condInd)

scaler = StandardScaler()
fourierCompX = scaler.fit_transform(fourierCompX)

largeSysName = []
for s in sysName:
    if s in sys2largeSys:
        largeSysName.append(sys2largeSys[s])
    else:
        largeSysName.append(s)

largeSysName = np.array(largeSysName)
largeSysUniq = np.unique(largeSysName)

# Manifold learning -----------------------------------------------------------
    
pca3 = PCA(n_components=3)
pca3_fit = pca3.fit_transform(fourierCompX)

pca2 = PCA(n_components=2)
pca2_fit = pca2.fit_transform(fourierCompX)

mds3 = MDS(n_components=3)
mds3_fit = mds3.fit_transform(fourierCompX)

mds2 = MDS(n_components=2)
mds2_fit = mds2.fit_transform(fourierCompX)

isomap3 = Isomap(n_components=3)
isomap3_fit = isomap3.fit_transform(fourierCompX)

isomap2 = Isomap(n_components=2)
isomap2_fit = isomap2.fit_transform(fourierCompX)

lle3 = LocallyLinearEmbedding(n_components=3)
lle3_fit = lle3.fit_transform(fourierCompX)

lle2 = LocallyLinearEmbedding(n_components=2)
lle2_fit = lle2.fit_transform(fourierCompX)

# -----------------------------------------------------------------------------

def wrapText(textList, w=40):
    
    return list(map(lambda t: '<br>'.join(wrap(t, w)), textList))


def create_manifold3(manifold_fit):
    
    manifold_data = []
    for i, lsys in enumerate(largeSysUniq):
        
        trace = go.Scatter3d(
            x=manifold_fit[largeSysName==lsys, 0],
            y=manifold_fit[largeSysName==lsys, 1],
            z=manifold_fit[largeSysName==lsys, 2],
            text=wrapText((np.array(condName)[largeSysName==lsys])),
            customdata=condInd[largeSysName==lsys],
            name=lsys,
            mode='markers',
            marker=dict(
                size=6,
                line=dict(
                    width=0.0
                ),
                opacity=0.8,
                color=palette[i]
            )
            
        )
                
        manifold_data.append(trace)
    
    manifold_layout = go.Layout(
        showlegend=True,
        legend=dict(font=dict(size=9),
                    orientation='h'),
        height=500,
        title='Low-dimensional embedding of harmonics'
    )
    
    return manifold_data, manifold_layout
    

def create_manifold2(manifold_fit):
    
    manifold_data = []
    for i, lsys in enumerate(largeSysUniq):
        
        trace = go.Scatter(
            x=manifold_fit[largeSysName==lsys, 0],
            y=manifold_fit[largeSysName==lsys, 1],
            text=wrapText((np.array(condName)[largeSysName==lsys])),
            customdata=condInd[largeSysName==lsys],
            name=lsys,
            mode='markers',
            marker=dict(
                size=8,
                line=dict(
                    width=0.0
                ),
                opacity=0.8,
                color=palette[i]
            )
            
        )
                
        manifold_data.append(trace)
    
    manifold_layout = go.Layout(
        showlegend=True,
        legend=dict(font=dict(size=9),
                    orientation='h'),
        height=500,
        title='Low-dimensional embedding of harmonics'
    )
    
    return manifold_data, manifold_layout


dim3manifolds = {'Principal Component Analysis (PCA)': create_manifold3(pca3_fit),
                 'Multi-dimensional Scaling (MDS)': create_manifold3(mds3_fit),
                 'Isomap': create_manifold3(isomap3_fit),
                 'Locally Linear Embedding (LLE)': create_manifold3(lle3_fit)}

dim2manifolds = {'Principal Component Analysis (PCA)': create_manifold2(pca2_fit),
                 'Multi-dimensional Scaling (MDS)': create_manifold2(mds2_fit),
                 'Isomap': create_manifold2(isomap2_fit),
                 'Locally Linear Embedding (LLE)': create_manifold2(lle2_fit)}

# -----------------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

available_algorithms = ['Principal Component Analysis (PCA)',
                        'Multi-dimensional Scaling (MDS)',
                        'Isomap',
                        'Locally Linear Embedding (LLE)']

app.layout = html.Div([
        
    html.Div([
        html.H2('Seasonalities of diseases',
                style={
                    'width': '100%',
                    'bottom': '2rem',
                    'left': '4rem',
                    'position': 'relative',
                    'display': 'inline',
                    'font-size': '6.0rem'
                })
    ]),
        
    html.Div([
        html.Div([
            html.P('HOVER over a condition in the embedding graph to see its seasonality. '
                   'SCROLL to zoom in and out the 3D embedding graph.'),
            html.P('HOLD left mouse button and DRAG to rotate the 3D embedding graph. '
                   'HOLD right mouse button and DRAG to pan the 3D embedding graph.'),
        ]),
        dcc.Dropdown(
            id='manifold algorithm',
            options=[{'label': i, 'value': i} for i in available_algorithms],
            value='Isomap',
            style={'width': '50rem'}
        ),
        dcc.RadioItems(
            id='dimensionality',
            options=[{'label': i, 'value': i} for i in ['2D', '3D']],
            value='3D',
            labelStyle={'display': 'inline-block'}
        )
    ],
    style={'width': '100%', 'display': 'inline-block', 'padding-left': '4rem'}),
    
    html.Div([
        dcc.Graph(
            id='manifold',
            hoverData={'points': [{'customdata': 0}]}
        )
    ],
    style={'display': 'inline-block',
           'width': '49%'}),    
    
    html.Div([
        dcc.Graph(id='seasonality'),
    ], style={'display': 'inline-block', 
              'width': '49%'}),

    
], style={'margin': '5rem'})
    
@app.callback(
    dash.dependencies.Output('manifold', 'figure'),
    [dash.dependencies.Input('manifold algorithm', 'value'),
     dash.dependencies.Input('dimensionality', 'value')])
def update_manifold(manifold_algorithm, dim):

    if dim == '2D':
        
        manifold_data, manifold_layout = dim2manifolds[manifold_algorithm]
        
    else:
        
        manifold_data, manifold_layout = dim3manifolds[manifold_algorithm]
        
    return {'data': manifold_data, 'layout': manifold_layout}
        
    

@app.callback(
    dash.dependencies.Output('seasonality', 'figure'),
    [dash.dependencies.Input('manifold', 'hoverData')])
def update_seasonality(hoverData):
    
    condition_index = hoverData['points'][0]['customdata']
    hpd = seasonalityPlotData[condition_index]['hpd']
    mean = seasonalityPlotData[condition_index]['mean']
    thisCondName = '<br>'.join(wrap(condName[condition_index], 60))
    
    upper_bound = go.Scatter(
        y=hpd[:, 1],
        name='Upper bound',
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.15)',
        fill='tonexty')
    
    trace = go.Scatter(
        y=mean,
        name='Mean',
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.15)',
        fill='tonexty')
    
    lower_bound = go.Scatter(
        y=hpd[:, 0],
        name='Lower bound',
        marker=dict(color='rgba(68, 68, 68, 0.15)'),
        line=dict(width=0),
        mode='lines')

    data = [lower_bound, trace, upper_bound]
    
    fig = {'data': data, 
           'layout': go.Layout(
                   yaxis=dict(title='DR seasonal fluctuation (95% C.I.)',
                              showexponent='last',
                              exponentformat='power'),
                   xaxis=dict(tickvals=monthTicks,
                              ticktext=MONTHS),
                   height=500,
                   title=thisCondName,
                   showlegend = False)}
                
    return fig
    
if __name__ == '__main__':
    app.run_server(debug=False)
    
    