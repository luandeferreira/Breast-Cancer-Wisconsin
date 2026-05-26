# Instale os pacotes descomentando a linha abaixo caso seja a primeira execução
# install.packages(c("caret", "mlbench", "e1071", "rpart", "randomForest", "nnet"))

suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(mlbench))

cat("--- Carregando Base de Dados Breast Cancer Wisconsin ---\n")
data(BreastCancer)

# Removendo NAs (o dataset do mlbench possui alguns valores faltantes) e a coluna ID
df <- na.omit(BreastCancer)
df <- df[, -1]

# Convertendo colunas para numéricas para permitir a padronização
for(i in 1:9) {
  df[, i] <- as.numeric(as.character(df[, i]))
}

# Fixando a semente aleatória para reprodutibilidade
set.seed(42)

# Divisão de Treino e Teste (80/20)
trainIndex <- createDataPartition(df$Class, p = .8, list = FALSE, times = 1)
dfTrain <- df[trainIndex,]
dfTest  <- df[-trainIndex,]

# Configurando o controle do treinamento (Cross-Validation simples para otimização interna)
fitControl <- trainControl(method = "cv", number = 5)

# Lista para armazenar os modelos treinados
modelos <- list()

cat("Treinando modelos...\n")

# 1. Naive Bayes
set.seed(42)
modelos[["Naive Bayes"]] <- train(Class ~ ., data = dfTrain, method = "naive_bayes", preProcess = c("center", "scale"), trControl = fitControl)

# 2. Árvore de Decisão
set.seed(42)
modelos[["Árvore de Decisão"]] <- train(Class ~ ., data = dfTrain, method = "rpart", preProcess = c("center", "scale"), trControl = fitControl)

# 3. Random Forest
set.seed(42)
modelos[["Random Forest"]] <- train(Class ~ ., data = dfTrain, method = "rf", preProcess = c("center", "scale"), trControl = fitControl)

# 4. Redes Neurais
set.seed(42)
modelos[["Redes Neurais"]] <- train(Class ~ ., data = dfTrain, method = "nnet", preProcess = c("center", "scale"), trControl = fitControl, trace = FALSE)

cat("\n--- Resultados da Avaliação ---\n\n")

# Avaliação no conjunto de teste
for(nome in names(modelos)) {
  predicoes <- predict(modelos[[nome]], newdata = dfTest)
  
  # Calculando a matriz de confusão e métricas
  cm <- confusionMatrix(predicoes, dfTest$Class, positive = "malignant")
  
  cat(paste("Modelo:", nome, "\n"))
  cat(sprintf("Acurácia:  %.4f\n", cm$overall["Accuracy"]))
  cat(sprintf("Precisão:  %.4f\n", cm$byClass["Pos Pred Value"]))
  cat(sprintf("Revocação: %.4f\n", cm$byClass["Sensitivity"]))
  cat(sprintf("F1-Score:  %.4f\n", cm$byClass["F1"]))
  cat("Matriz de Confusão:\n")
  print(cm$table)
  cat("------------------------------\n")
}
