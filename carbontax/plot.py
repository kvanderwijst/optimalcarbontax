## Import packages

import plotly.offline as pyo
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as pyt

import datetime
import numpy as np

def plotlyInteractive():
    pyo.init_notebook_mode(connected=True)



def plotStateAndControl (outputValues, showProxy=True, interactive=False):
    tValues = outputValues['meta']['tValuesDate']
    
    fig = pyt.make_subplots(
        rows=2, cols=1,
        print_grid=False,
        subplot_titles=('Emissions (rel. to 2015)','Carbon tax'),
        shared_xaxes=True
    )

    fig.append_trace(go.Scatter(x=tValues, y=outputValues['meta']['baseline'], name='Baseline'), 1, 1)
        
    fig.append_trace(go.Scatter(x=tValues, y=outputValues['emissions'], mode='lines', name='Abatement scenario'), 1, 1)
    fig.append_trace(go.Scatter(x=tValues, y=outputValues['price'], name='Carbon tax'), 2, 1)
    
    if showProxy is not False:
        fig.append_trace(go.Scatter(x=tValues, y=outputValues['proxyPrice'], name='Proxy Hotelling', line={'color': '#BBB'}), 2, 1)
        fig.append_trace(go.Scatter(x=tValues, y=outputValues['proxyEmissions'], name='Proxy Hotelling Emissions', line={'color': '#BBB'}), 1, 1)

    fig['layout'].update(
        yaxis1={'range': [min(0,np.amin(outputValues['emissions'])-0.1), outputValues['meta']['baseline'][-1]*1.05], 'title': '(rel. to 2015)'},
        yaxis2={'title': '( $ / tCO2e )', 'range': [0, min(np.amax(outputValues['price'])*1.2, np.amax(outputValues['proxyPrice']))]},
        height=700
    )

    if (interactive):
        pyo.iplot(fig)
    else:
        pyo.plot(fig)