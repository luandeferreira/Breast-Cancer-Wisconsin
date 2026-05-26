# Entendendo o Projeto: Classificação de Câncer de Mama

Este projeto realiza um estudo comparativo entre algoritmos de *Machine Learning* para classificar tumores de mama como Benignos ou Malignos. O objetivo atende aos requisitos da disciplina, implementando e comparando o desempenho de quatro algoritmos clássicos em Python e R.

## 1. O Problema e a Base de Dados
Utilizamos o **Breast Cancer Wisconsin (Diagnostic)**, um dataset público amplamente reconhecido na literatura de Ciência de Dados. Ele contém características computadas a partir de imagens digitalizadas de biópsias de mama (como raio, textura, perímetro e área celular).
* **Tipo de Problema:** Classificação Supervisionada Binária.
* **Por que essa base?** É um problema real, com impacto direto na área da saúde. A base possui excelente qualidade, não exige tratamentos complexos de *outliers* e possui atributos puramente numéricos contínuos, facilitando a implementação de algoritmos baseados em distância (como Redes Neurais).

## 2. Algoritmos Implementados
O projeto implementa rigorosamente os quatro modelos exigidos:
1. **Naive Bayes:** Um modelo probabilístico simples e rápido, baseado no Teorema de Bayes, que assume que as variáveis explicativas (as medidas da célula) são independentes entre si.
2. **Árvore de Decisão:** Um modelo simbólico que cria uma série de regras do tipo "Se-Então" baseadas nos atributos da célula para chegar a um diagnóstico. É altamente interpretável.
3. **Random Forest:** Um método de *ensemble* que treina dezenas de Árvores de Decisão e realiza uma "votação" para o diagnóstico final. Reduz o sobreajuste (overfitting) da árvore individual e geralmente entrega excelente desempenho.
4. **Redes Neurais (MLP - Multi-Layer Perceptron):** Um modelo bio-inspirado com camadas ocultas de neurônios matemáticos que aprendem representações não-lineares complexas dos dados celulares.

## 3. Pré-processamento e Metodologia Experimental
* **Divisão dos Dados:** A base foi dividida em 80% para treino e 20% para teste. Fixamos uma semente aleatória (`seed 42`) para garantir reprodutibilidade.
* **Padronização:** As variáveis foram escalonadas (Z-score). Isso é crucial para algoritmos sensíveis à escala, garantindo que variáveis com grandezas maiores (como 'área') não dominem erroneamente os cálculos da Rede Neural em relação a variáveis com grandezas menores (como 'suavidade').

## 4. Como Ler os Resultados na Defesa
Para o contexto médico, **Acurácia não é a métrica mais importante**. A defesa do trabalho deve focar na **Revocação (Recall / Sensibilidade)**. 
* **Falso Positivo:** Dizer que um paciente tem câncer quando não tem (gera susto e exames extras).
* **Falso Negativo (O Pior Erro):** Dizer que um paciente está saudável quando ele tem câncer (pode custar uma vida).
A **Revocação** mede justamente a capacidade do modelo de encontrar **todos** os casos que realmente são malignos, evitando os Falsos Negativos. O **F1-Score** também é analisado por trazer o equilíbrio harmônico entre a Precisão e a Revocação.
