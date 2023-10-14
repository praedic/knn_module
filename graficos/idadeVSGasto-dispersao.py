import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../dataset_comercial.csv')

sns.scatterplot(data=data, x='Idade', y='Nível de Gasto', hue='Comprou', palette='coolwarm')
plt.title('Distribuição de Idade vs. Nível de Gasto com base em Compra')
plt.savefig('distribuicao_idade_vs_gasto.png')
plt.show()
