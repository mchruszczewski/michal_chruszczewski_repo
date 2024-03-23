
#DATA PROCESSING FUNCTION

process_data <- function(df,train=TRUE) {
  current_year <- 2024
  
  # normalizing variables
  df <- df %>%
    mutate(LicenseYear = round(LicenseYear),
           EngineCap = round(EngineCap),
           BirthYear = current_year - round(BirthYear),
           CarYear = current_year - CarYear)
  
  # setting factors
  if (train== TRUE){
    df <- df %>%
      mutate(across(c(Sex, Accidents, CarBrand, CarType, NextAccident), as.factor))
  }else{
    df <- df %>%
      mutate(across(c(Sex, Accidents, CarBrand, CarType), as.factor))
  }
  
  #imputation
  
  # factors
  getMode <- function(x) {
    uniqx <- unique(x)
    uniqx[which.max(tabulate(match(x, uniqx)))]
  }
  for(i in which(sapply(df, is.factor))) {
    mode_value <- getMode(df[,i][!is.na(df[,i])])
    df[,i][is.na(df[,i])] <- mode_value
  }
  
  #numeric
  for(i in which(sapply(df, is.numeric))) {
    df[,i][is.na(df[,i])] <- median(df[,i], na.rm = TRUE)
  }
  
  #removing outliers
  if (train== TRUE){
    nazwy_kolumn_outlier<- c('BirthYear', 'CarYear', 'CarEngine', 'EngineCap', 'AssistanceYears')
    outlier_function <- function(df,kolumna){
      Q1 <- quantile(df[[kolumna]], 0.25)
      Q3 <- quantile(df[[kolumna]], 0.75)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      df <- df[df[[kolumna]] > lower_bound & df[[kolumna]] < upper_bound, ]
      return(df)
    }
    for (kolumna in nazwy_kolumn_outlier){
      df <- outlier_function(df,kolumna)
    }
  }
  
  #grouping some brands and accidents
  
  df$CarBrand <- ifelse(df$CarBrand %in% c('6', '9'), '6', df$CarBrand)
  df$CarBrand <- ifelse(df$CarBrand %in% c('7', '8'), '7', df$CarBrand)
  df$Accidents <- ifelse(df$Accidents %in% c('6', '7'), '6', df$Accidents)
  
  # log-transforming the data
  df$BirthYear <- log(df$BirthYear + 1)
  df$LicenseYear <- log(df$LicenseYear + 1)
  df$CarYear <- log(df$CarYear + 1)
  df$CarEngine <- log(df$CarEngine + 1)
  df$EngineCap <- log(df$EngineCap + 1)
  df$CarValue <- log(df$CarValue + 1)
  df$AssistanceYears <- log(df$AssistanceYears + 1)
  
  #Returning train and test datasets
  
  if (train==TRUE){
    #DATA SET SPLIT ON TRAINING AND TESTING FUNCTION:
    SplitDataSet<-function(data.set,training.fraction) {
      random.numbers<-sample.int(nrow(data.set))
      quantiles<-quantile(random.numbers,probs=c(0,training.fraction,1))
      split.labels<-cut(random.numbers,quantiles,include.lowest=T,labels=c("training","test"))
      return(split(data.set,split.labels))
    }
    split.dataset<-SplitDataSet(df,0.7)
    return(split.dataset)
    }else {return(df)}
}


ovun <- function(df) {
  set.seed(1)  
  
  df_0 <- df[df$NextAccident == 0, ]
  df_1 <- df[train.set$NextAccident == 1, ]

  n_times <- nrow(df_0) / nrow(df_1)
  
  df_1_oversampled <- df_1[sample(1:nrow(df_1), size = nrow(df_0), replace = TRUE), ]
  
  df_oversampled <- rbind(df_0, df_1_oversampled)
  
  return(df_oversampled)
}
