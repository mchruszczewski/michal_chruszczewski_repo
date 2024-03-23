#1. Create plots

create.plots <- function(df){
  plot.list <- list()
  df <- na.omit(df)
  
  #A. Correlation
  cor_matrix <- cor(df, method = "spearman")
  correlation<- corrplot(cor_matrix,method='ellipse')
  plot.list<- correlation
  
  #B. Outliers- boxplots
  
  for (i in 1:(ncol)(df)) {
    new_element<-boxplot(df[[i]],main= names(df)[i] ,ylab=names(df)[i])
    plot.list<- append(plot.list, new_element)
  }
  
  #C. Histograms
  
  for (i in 1:(ncol)(df)){
    new_element<- hist(df[[i]], main= names(df)[i] ,xlab=names(df)[i], ylab="")
    plot.list<- append(plot.list, new_element)
  }
  
  return(plot.list)
}


#2. Create tables to inspect data

create.tables <- function(df){
  tables.list <- list()
  
  #A. Create contigent tables
  
  factor_columns <- c('Sex', 'CarType', 'Accidents', 'CarBrand','NextAccident')
  contingent_tables <- list()
  for (col in factor_columns) {
    if(col != "NextAccident") { # Pominięcie, gdyż nie chcemy porównywać kolumny z samą sobą
      table <- CrossTable(df$NextAccident, df[[col]],
                          prop.c= FALSE, prop.chisq= FALSE, prop.t=FALSE,
                          dnn = c('NextAccident', col)) # Dodatkowe opcje dla czytelności
      contingent_tables[[col]] <- table
    }
  }
  tables.list<- contingent_tables
  
  
  return(tables.list)
}

  




