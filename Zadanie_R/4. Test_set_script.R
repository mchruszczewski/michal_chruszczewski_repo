#2. Model testing

test.models <- function(df, model, type, cutoff){
  if (type %in% c('link', 'response', 'terms')){
    predicted_values <- predict(model, newdata = df, type = type)
  } else {
    predicted_values <- predict(model, newdata = df, type =type)[,2]
  }
  
  cutoff<- cutoff$optimal_cutoff
  actual_classes <- df$NextAccident
  predicted_classes <- ifelse(predicted_values > cutoff, "1", "0")
  predicted_classes <- factor(predicted_classes)
  TN <- sum(actual_classes == "0" & predicted_classes == "0")
  FN <- sum(actual_classes == "1" & predicted_classes == "0")
  profits <- (TN * 1100) - (FN * 5500)
  confMat <- confusionMatrix(predicted_classes, actual_classes)
  
  return(list(profit = profits, name = class(model)[1], confusionmatrix= confMat))
}


compare.models <- function(result1, result2) {
  if (result1$profit > result2$profit){
    return(result1$name)
  } else {
    return(result2$name)
  }
}


new.set.test <- function(df, compare, model1, model2, cutoff1, cutoff2){
  
  if(compare=="randomForrest"){
    cutoff1<- cutoff1$optimal_cutoff
    prediction<- predict(model1, newdata= df, type= 'prob')
    predicted_classes <- ifelse(prediction > cutoff1, "1", "0")
    df$NextAccident <- predicted_classes
    return (df)
  }else{cutoff2<- cutoff1$optimal_cutoff
    prediction <-predict(model2, newdata= df, type='response')
    predicted_classes <- ifelse(prediction > cutoff2, "1", "0")
    df$NextAccident <- predicted_classes
    return (df)}
}


