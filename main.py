import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("--- Carregando Arquivo wdbc.data ---")
    
    # 1. Lê o arquivo local, igual fizemos no R (sem cabeçalho)
    try:
        df = pd.read_csv('wdbc.data', header=None)
    except FileNotFoundError:
        print("Erro: Arquivo 'wdbc.data' não encontrado. Verifique se está na mesma pasta.")
        return

    # 2. Entendendo o df:
    # Coluna 0: ID do paciente (descartar)
    # Coluna 1: Diagnóstico (M ou B)
    # Colunas 2 em diante: Características numéricas
    
    # Separando as variáveis (X) do alvo (y)
    X = df.iloc[:, 2:] # Pega da coluna 2 até o final
    y_raw = df.iloc[:, 1]  # Pega apenas a coluna 1
    
    # Mapeando M (Maligno) para 1 e B (Benigno) para 0. 
    # Isso alinha a classe positiva com o foco médico (encontrar o câncer)
    y = y_raw.map({'M': 1, 'B': 0})

    # Divisão de Treino e Teste (80/20) com semente fixa para reprodutibilidade
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pré-processamento: Padronização dos dados (Z-score)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dicionário com os 4 modelos obrigatórios
    modelos = {
        "Naive Bayes": GaussianNB(),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Redes Neurais (MLP)": MLPClassifier(random_state=42, max_iter=500)
    }

    print("\n--- Resultados da Avaliação ---\n")
    
    for nome, modelo in modelos.items():
        # Treinamento
        modelo.fit(X_train_scaled, y_train)
        
        # Predição
        y_pred = modelo.predict(X_test_scaled)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Modelo: {nome}")
        print(f"Acurácia:  {acc:.4f}")
        print(f"Precisão:  {prec:.4f}")
        print(f"Revocação: {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        # A matriz do sklearn sai em formato um pouco diferente do R, mas a leitura é a mesma
        print(f"Matriz de Confusão:\n{cm}") 
        print("-" * 30)

if __name__ == "__main__":
    main()
