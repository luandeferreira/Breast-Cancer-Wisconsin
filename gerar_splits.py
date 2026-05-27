import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar os dados originais
colunas = ['id', 'diagnostico'] + [f'feat_{i}' for i in range(1, 31)]
df = pd.read_csv("wdbc.data", header=None, names=colunas)

# 2. Limpeza e conversão do alvo (M = 1, B = 0)
df = df.drop(columns=['id'])
df['diagnostico'] = df['diagnostico'].map({'M': 1, 'B': 0})

# 3. Separar as features (X) do alvo (y)
X = df.drop(columns=['diagnostico'])
y = df['diagnostico']

# 4. Fazer o split com semente fixa (80% treino, 20% teste)
# O stratify=y garante a mesma proporção de casos malignos e benignos no treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Juntar e salvar em CSV
treino = pd.concat([y_treino, X_treino], axis=1)
teste = pd.concat([y_teste, X_teste], axis=1)

treino.to_csv("treino.csv", index=False)
teste.to_csv("teste.csv", index=False)

print("Sucesso! Arquivos 'treino.csv' e 'teste.csv' gerados.")
