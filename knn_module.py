import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import plotly.graph_objects as go

def generate_plotly_figure(knn, ct, highlighted_point=None):
    data = pd.read_csv('dataset_comercial.csv')

    # Gerar regiões de decisão
    xx, yy, Z = plot_decision_boundaries(knn, ct, min(data['Idade']), max(data['Idade']), min(data['Nível de Gasto']), max(data['Nível de Gasto']))
    
    # Crie um gráfico de dispersão usando Plotly
    fig = go.Figure()

    # Adicione as regiões de decisão
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, showscale=False, opacity=0.4, colorscale='Viridis'))
    
    # Adicione os pontos de dados
    fig.add_trace(go.Scatter(x=data['Idade'][data['Comprou'] == 0], y=data['Nível de Gasto'][data['Comprou'] == 0], mode='markers', marker=dict(color='blue'), showlegend=True, name="Não Comprou"))
    fig.add_trace(go.Scatter(x=data['Idade'][data['Comprou'] == 1], y=data['Nível de Gasto'][data['Comprou'] == 1], mode='markers', marker=dict(color='yellow'), showlegend=True, name="Comprou"))
    
    # Adicione o ponto destacado, se fornecido
    if highlighted_point:
        fig.add_trace(
            go.Scatter(x=[highlighted_point[0]], 
                       y=[highlighted_point[1]], 
                       mode='markers',
                       marker=dict(color='red', size=10, line=dict(width=2, color='DarkRed')),
                       name='Ponto Destacado')
        )

    fig.update_layout(title='Regiões de Decisão com KNN',
                      xaxis_title="Idade",
                      yaxis_title="Nível de Gasto",
                      legend_title="Legendas")
    
    # Converta a figura do Plotly em HTML
    plotly_html = fig.to_html(full_html=False)
    
    return plotly_html


def plot_decision_boundaries(knn, ct, min_age, max_age, min_spend, max_spend):
    # Criar uma malha de pontos

    age_range = np.linspace(min_age, max_age, 100)
    spend_range = np.linspace(min_spend, max_spend, 100)
    xx, yy = np.meshgrid(age_range, spend_range)

    default_time_on_site = np.zeros_like(xx.ravel())
    default_gender = np.array(['Masculino'] * len(xx.ravel()))  # Valor padrão para o gênero

    # Preparar os pontos da malha para a previsão
    mesh_samples = np.column_stack((xx.ravel(), default_gender, yy.ravel(), default_time_on_site))
    mesh_encoded = ct.transform(mesh_samples)

    # Fazer previsões para cada ponto
    Z = knn.predict(mesh_encoded)
    Z = Z.reshape(xx.shape)
    
    return xx, yy, Z

def train_knn(k=5):
    # Carregar os dados
    data = pd.read_csv('dataset_comercial.csv')

    # Lidar com valores ausentes, se necessário
    data.dropna(inplace=True)

    # Dividir dados em recursos (X) e rótulos (y)
    X = data[['Idade', 'Gênero', 'Nível de Gasto', 'Tempo no Site (min)']]
    y = data['Comprou']

        # Criar um transformador de colunas para tratar as características numéricas e categóricas
    ct = ColumnTransformer(
        [("scaler", StandardScaler(), [0, 2, 3]),  # Índices das colunas numéricas
        ("encoder", OneHotEncoder(drop='first'), [1])]  # Índice da coluna 'Gênero'
    )

    X = ct.fit_transform(X)
    
    # Dividir em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    return knn, ct, X_test, y_test

def predict_sample(knn, ct, sample):
    sample_array = np.array(sample).reshape(1, -1)
    sample_encoded = ct.transform(sample_array)
    prediction = knn.predict(sample_encoded)
    return prediction[0]

def evaluate_model(knn, X_test, y_test):
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def find_best_k(X, y, max_k=80):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    accuracies = []
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies
if __name__ == "__main__":
    data = pd.read_csv('dataset_comercial.csv')
    X = data[['Idade', 'Gênero', 'Nível de Gasto', 'Tempo no Site (min)']]
    y = data['Comprou']
    ct = ColumnTransformer(
        [("scaler", StandardScaler(), [0, 2, 3]),
        ("encoder", OneHotEncoder(drop='first'), [1])]
    )
    X = ct.fit_transform(X)
    
    accuracies = find_best_k(X, y)
    best_k = accuracies.index(max(accuracies)) + 1
    print(f"Melhor valor de k: {best_k} com acurácia de {max(accuracies):.4f}")