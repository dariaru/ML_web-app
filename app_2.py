import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1(children='Прогноз переливания крови', className="header-title1"),
    dcc.Slider(
        id='cv-slider',
        min=1,
        max=10,
        step=1,
        value=5,
        marks={i:str(i) for i in range(1,11)}
    ),
    dcc.Slider(
        id='knn-slider',
        min=10,
        max=60,
        step=5,
        value=5,
        marks={i:str(i) for i in range(1,61)}
    ),
    dcc.Graph(id='scores-graphic'),
    html.Div(id='scores-output')
])

def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('transfusion.csv', sep=",")
    return data

data = load_data()
X=data[['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)','Time (months)']]
y=data['whether he/she donated blood in March 2007']
data_len = data.shape[0]

@app.callback(
    Output(component_id='scores-graphic', component_property='figure'),
    Output(component_id='scores-output', component_property='children'),
    Input(component_id='knn-slider', component_property='value'),
    Input(component_id='cv-slider', component_property='value')
)
def update_knn_slider(knn_slider, cv_slider):
    scores = cross_val_score(RandomForestClassifier(n_estimators=knn_slider), X, y, scoring='accuracy', cv=cv_slider)
    #Формирование графика
    fig = px.bar(x=list(range(1, len(scores)+1)), y=scores)
    #Вывод среднего значения
    mean_text = 'Усредненное значение accuracy по всем фолдам - {}'.format(np.mean(scores))
    return fig, mean_text

if __name__ == '__main__':
    app.run_server(debug=True)