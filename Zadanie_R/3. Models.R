#1. Creating GLM model
glm.model <- function(df){
  model.glm <- glm(NextAccident~ .,data = train.set, family = binomial)
  return(model.glm)
}

#2. Creating Random Forrest model
random.forrest <- function(df) {
  train.label <- df$NextAccident
  train.label <- as.factor(train.label)
  df <- df[ , !(names(df) %in% c("NextAccident"))]
  model <- randomForest(x = df, y = train.label, ntree = 500)
  
  return(model)
}

# 2. Cut-off optimisation

cutOff.optim <- function(df, model,type) {
  
  if (type %in% c('link', 'response', 'terms')){
    predicted_values <- predict(model, newdata = df, type = type)
  }else{predicted_values <- predict(model, newdata = df, type = type)[,2]}
  
  CutOff <- seq(from = 0.0, to = 1, by = 0.02)
  profits <- numeric(length(CutOff))
  actual_classes <- df$NextAccident
  for(i in seq_along(CutOff)) {
    predicted_classes <- ifelse(predicted_values > CutOff[i], "1", "0")
    TN <- sum(actual_classes == "0" & predicted_classes == "0")
    FN <- sum(actual_classes == "1" & predicted_classes == "0")
    profits[i] <- (TN * 1100) - (FN * 5500)
  }
  optimal_index <- which.max(profits)
  optimal_cutoff <- CutOff[optimal_index]
  max_profit <- profits[optimal_index]
  actual_classes <- factor(actual_classes, levels = c("0", "1"))
  predicted_classes <- factor(predicted_values > optimal_cutoff, levels = c(FALSE, TRUE), labels = c("0", "1"))
  confMat <- confusionMatrix(predicted_classes, actual_classes)
  
  cat("Optimal CutOff:", optimal_cutoff, "max profit", max_profit, "\n")
  
  
  
  return(list(optimal_cutoff = optimal_cutoff, max_profit = max_profit, confusion_matrix=confMat ))
}




