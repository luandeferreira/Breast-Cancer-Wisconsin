# ---------------------------------------------------------
# VERIFICAÇÃO E INSTALAÇÃO AUTOMÁTICA DE PACOTES
# ---------------------------------------------------------
pacotes_necessarios <- c("caret", "e1071", "rpart", "randomForest", "nnet", "naivebayes")

# Verifica quais pacotes da lista não estão instalados na máquina
pacotes_faltantes <- pacotes_necessarios[!(pacotes_necessarios %in% installed.packages()[,"Package"])]

# Se faltar algum, o R instala automaticamente sem você precisar digitar nada
if(length(pacotes_faltantes) > 0) {
  cat("Instalando pacotes faltantes na máquina...\n")
  install.packages(pacotes_faltantes, dependencies = TRUE, repos = "https://cloud.r-project.org")
}
# ---------------------------------------------------------

# Carrega a biblioteca principal
library(caret)

cat("--- Carregando Arquivos de Treino e Teste ---\n")

# 1. Carregar os dados já divididos (Gerados pelo Python)
treino <- read.csv("treino.csv")
teste <- read.csv("teste.csv")

# O R exige que a variável alvo seja um "Fator" (Factor) para classificação.
# Como o Python salvou como 0 e 1, vamos renomear para ficar claro nas métricas:
treino$diagnostico <- factor(treino$diagnostico, levels = c(0, 1), labels = c("Benigno", "Maligno"))
teste$diagnostico <- factor(teste$diagnostico, levels = c(0, 1), labels = c("Benigno", "Maligno"))

# 2. Padronização (O caret calcula no treino e aplica no teste)
preprocessParams <- preProcess(treino[, -1], method = c("center", "scale"))
treino_scaled <- predict(preprocessParams, treino[, -1])
teste_scaled <- predict(preprocessParams, teste[, -1])

# Juntar a coluna alvo de volta aos dados padronizados
treino_final <- data.frame(diagnostico = treino$diagnostico, treino_scaled)
teste_final <- data.frame(diagnostico = teste$diagnostico, teste_scaled)

# Função para facilitar a impressão
avaliar_modelo_R <- function(nome, modelo, teste_data) {
  cat("\n---", nome, "---\n")
  previsoes <- predict(modelo, teste_data)
  # positive = "Maligno" garante que a métrica foca na detecção do câncer
  cm <- confusionMatrix(previsoes, teste_data$diagnostico, positive = "Maligno")
  
  print(cm$table)
  cat("Acurácia :", round(cm$overall['Accuracy'], 4), "\n")
  cat("Precisão :", round(cm$byClass['Pos Pred Value'], 4), "\n")
  cat("Revocação:", round(cm$byClass['Sensitivity'], 4), "\n")
  cat("F1-Score :", round(cm$byClass['F1'], 4), "\n")
  cat("------------------------------\n")
}

# Configuração de validação cruzada para o caret (Ajuste de hiperparâmetros embutido)
train_control <- trainControl(method="cv", number=5)

# --- 3. Treinamento dos Modelos ---

# Naive Bayes
set.seed(42)
modelo_nb <- train(diagnostico ~ ., data = treino_final, method = "naive_bayes", trControl = train_control)
avaliar_modelo_R("Naive Bayes", modelo_nb, teste_final)

# Decision Tree (CART)
set.seed(42)
modelo_dt <- train(diagnostico ~ ., data = treino_final, method = "rpart", trControl = train_control)
avaliar_modelo_R("Decision Tree", modelo_dt, teste_final)

# Random Forest
set.seed(42)
modelo_rf <- train(diagnostico ~ ., data = treino_final, method = "rf", trControl = train_control)
avaliar_modelo_R("Random Forest", modelo_rf, teste_final)

# Redes Neurais
set.seed(42)
# trace = FALSE para não poluir o terminal com as iterações da rede neural
modelo_nn <- train(diagnostico ~ ., data = treino_final, method = "nnet", trControl = train_control, trace = FALSE)
avaliar_modelo_R("Rede Neural", modelo_nn, teste_final)
