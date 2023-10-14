import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../dataset_comercial.csv')

sns.countplot(data=data, x='Gênero', palette='pastel')
plt.title('Distribuição de Gênero')
plt.savefig('distribuicao_genero.png')
plt.show()

sns.countplot(data=data, x='Comprou', palette='muted')
plt.title('Distribuição de Compra')
plt.savefig('distribuicao_compra.png')
plt.show()