import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV # Biblioteca para o Bônus!

# 1. Carregar os dados já divididos e idênticos ao R
treino = pd.read_csv("treino.csv")
teste = pd.read_csv("teste.csv")

X_train = treino.drop(columns=['diagnostico'])
y_train = treino['diagnostico']
X_test = teste.drop(columns=['diagnostico'])
y_test = teste['diagnostico']

# 2. Padronização (O fit é feito APENAS no treino)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Função para imprimir os resultados
def avaliar_modelo(nome, y_true, y_pred):
    print(f"--- {nome} ---")
    print(f"Acurácia : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precisão : {precision_score(y_true, y_pred):.4f}")
    print(f"Revocação: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")
    print("Matriz de Confusão:\n", confusion_matrix(y_true, y_pred))
    print("-" * 30)

# --- 3. Treinamento dos Modelos ---

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
avaliar_modelo("Naive Bayes", y_test, nb.predict(X_test_scaled))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
avaliar_modelo("Decision Tree", y_test, dt.predict(X_test_scaled))

# --- BÔNUS: RANDOM FOREST COM GRIDSEARCH CV ---
print("\n[BÔNUS] Otimizando hiperparâmetros da Random Forest...\n")
rf = RandomForestClassifier(random_state=42)

# Definimos as combinações de parâmetros que o modelo vai testar
parametros_rf = {
    'n_estimators': [50, 100, 150],       # Número de árvores
    'max_depth': [None, 10, 20],          # Profundidade máxima
    'min_samples_split': [2, 5]           # Mínimo de amostras para dividir um nó
}

# O GridSearchCV vai testar todas as combinações acima usando validação cruzada (cv=5)
grid_rf = GridSearchCV(estimator=rf, param_grid=parametros_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)

print(f"Melhores parâmetros encontrados: {grid_rf.best_params_}")
avaliar_modelo("Random Forest (Otimizada)", y_test, grid_rf.predict(X_test_scaled))

# Redes Neurais (MLP)
mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
avaliar_modelo("Rede Neural (MLP)", y_test, mlp.predict(X_test_scaled))
