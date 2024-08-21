library(data.table)


rm(list=ls())
gc(full= TRUE)
setwd('~/Documents/Inżynieria Danych- Big Data/Projekt końcowy/R- case study')

df <- read.csv('PredMaint.csv', sep = ';')
data <- data.table::data.table(df)
data$Y= data$Y - 1 

table(data$Y)

#zmienne wielomianowe

library(fastDummies)

data.dum= dummy_cols(data, remove_selected_columns = TRUE, remove_first_dummy = T)

df.dum<- dummy_cols(df, remove_selected_columns = TRUE, remove_first_dummy = T)
#Linear Probability Model
pm.lm <- lm(Y~ ., data= data.dum, remove_selected_columns = TRUE, remove_first_dummy = T)
summary(pm.lm)
hist(pm.lm$fitted.values)

# GLM

pm.glm = glm(Y~ ., data = data.dum, family = binomial(link='logit'))
hist(pm.glm$fitted.values)
summary(pm.glm)


#ANN Fast Forward


library(neuralnet)

for (i in 1: ncol(df.dum)){
  
  x= df.dum[,i]
  x=(x-min(x))/(max(x)-min(x))
  df.dum[,i]=x
}
pm.ann <- neuralnet(Y ~ ., data = df.dum, hidden= c(3, 2), linear.output = F)
hist(pm.ann$net.result[[1]])
table(pm.ann$net.result[[1]])
