import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../dataset_comercial.csv')

sns.histplot(data['Idade'], kde=True, color='skyblue')
plt.title('Distribuição da Idade')
plt.savefig('distribuicao_idade.png')
plt.show()

sns.histplot(data['Nível de Gasto'], kde=True, color='salmon')
plt.title('Distribuição do Nível de Gasto')
plt.savefig('distribuicao_nivel_gasto.png')
plt.show()

sns.histplot(data['Tempo no Site (min)'], kde=True, color='green')
plt.title('Distribuição do Tempo no Site')
plt.savefig('distribuicao_tempo_site.png')
plt.show()

