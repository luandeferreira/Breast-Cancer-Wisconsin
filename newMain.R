# # Carrega a biblioteca necessária
# library(caret)
# 
# cat("--- Carregando Arquivo wdbc.data ---\n")
# 
# # 1. Lê o arquivo wdbc.data. Como ele NÃO tem cabeçalho, usamos header = FALSE.
# # O R vai chamar as colunas automaticamente de V1, V2, V3...
# df <- read.csv("wdbc.data", header = FALSE)
# 
# # 2. Entendendo o wdbc.data:
# # Coluna 1 (V1): ID do paciente
# # Coluna 2 (V2): Diagnóstico (M = Maligno, B = Benigno)
# # Colunas 3 a 32 (V3 a V32): As características da célula
# 
# # Removemos a coluna de ID (V1), pois identificador de paciente não serve para treinar modelo
# df <- df[, -1]
# 
# # Agora o Diagnóstico (antiga V2) virou a primeira coluna. 
# # Vamos renomeá-la para "diagnosis" para ficar fácil de referenciar
# colnames(df)[1] <- "diagnosis"
# 
# # O R exige que a variável alvo seja um "fator" (categoria) para fazer classificação
# df$diagnosis <- as.factor(df$diagnosis)
# 
# # Fixamos a semente para que o resultado seja o mesmo toda vez que você rodar (reprodutibilidade)
# set.seed(42)
# 
# # Divisão de Treino (80%) e Teste (20%)
# trainIndex <- createDataPartition(df$diagnosis, p = .8, list = FALSE, times = 1)
# dfTrain <- df[trainIndex,]
# dfTest  <- df[-trainIndex,]
# 
# # Configuração da Validação Cruzada (ajuda o modelo a não decorar os dados)
# fitControl <- trainControl(method = "cv", number = 5)
# 
# # Lista para guardar os 4 modelos exigidos
# modelos <- list()
# 
# cat("Treinando modelos...\n")
# 
# # 1. Naive Bayes
# set.seed(42)
# modelos[["Naive Bayes"]] <- train(diagnosis ~ ., data = dfTrain, method = "naive_bayes", preProcess = c("center", "scale"), trControl = fitControl)
# 
# # 2. Árvore de Decisão
# set.seed(42)
# modelos[["Árvore de Decisão"]] <- train(diagnosis ~ ., data = dfTrain, method = "rpart", preProcess = c("center", "scale"), trControl = fitControl)
# 
# # 3. Random Forest
# set.seed(42)
# modelos[["Random Forest"]] <- train(diagnosis ~ ., data = dfTrain, method = "rf", preProcess = c("center", "scale"), trControl = fitControl)
# 
# # 4. Redes Neurais (MLP simples)
# set.seed(42)
# modelos[["Redes Neurais"]] <- train(diagnosis ~ ., data = dfTrain, method = "nnet", preProcess = c("center", "scale"), trControl = fitControl, trace = FALSE)
# 
# cat("\n--- Resultados da Avaliação ---\n\n")
# 
# # Vamos testar cada modelo no conjunto de dados separados para teste
# for(nome in names(modelos)) {
#   predicoes <- predict(modelos[[nome]], newdata = dfTest)
#   
#   # Cria a matriz de confusão. Definimos positive = "M" porque acertar o Maligno é o nosso foco.
#   cm <- confusionMatrix(predicoes, dfTest$diagnosis, positive = "M")
#   
#   # Imprime as métricas no console
#   cat(paste("Modelo:", nome, "\n"))
#   cat(sprintf("Acurácia:  %.4f\n", cm$overall["Accuracy"]))
#   cat(sprintf("Precisão:  %.4f\n", cm$byClass["Pos Pred Value"]))
#   cat(sprintf("Revocação: %.4f\n", cm$byClass["Sensitivity"]))
#   cat(sprintf("F1-Score:  %.4f\n", cm$byClass["F1"]))
#   cat("Matriz de Confusão:\n")
#   print(cm$table)
#   cat("------------------------------\n")
# }
