library(dplyr)
library(caret)


## Useful functions

change_char_to_factor <- function(df){
  for(col in names(df)){
    if(class(df[, col]) == "character" ){
      if(sum(is.na(df[,col])) == 0 ){
        df[, col] <- as.factor(df[, col])
      } else {
        df[, col] <- NULL
      }
    }
  }
  return (df)
}

pre_process <- function(df, method){
  pre_model <- preProcess(df, method = method) 
  pre_data <- predict(pre_model, df)
  
  return (pre_data)
}

evalute_model <- function(model, data, y){
  prediction <- predict(model, data)
  pred_df <- data.frame(obs = y, pred=prediction)
  return (pred_df)
}



## Data Exploration
train <- read.csv("train.csv",stringsAsFactors = FALSE)
glimpse(train)
summary(train)
sum(is.na(train))

## Data cleansing

sum(is.na(train$Alley))
train$Alley <- NULL

sum(is.na(train$PoolQC))
train$PoolQC <- NULL

sum(is.na(train$Fence))
train$Fence <- NULL

sum(is.na(train$MiscFeature))
train$MiscFeature <- NULL

train$Id<- NULL

train_clean <- change_char_to_factor(train)

glimpse(train_clean)
sum(is.na(train))


## Split the data into training and testing datasets

set.seed(42)
rows <- sample(nrow((train_clean)))
training <- train_clean[rows,]
split <- round(nrow(train_clean)*.80)
train_data <- training[1:split, ]
test_data <- training[(split + 1):nrow(training), ]
nrow(train_data) + nrow(test_data)


## Data preprocessing
x_train <- train_data[-ncol(train_data)]
y_train <- train_data %>% select(SalePrice)
x_test <- test_data[-ncol(test_data)]
y_test <- test_data %>% select(SalePrice)

### Data preprocessing - Median imputation 
x_train_mi <- pre_process(x_train, "medianImpute")
x_test_mi <- pre_process(x_test, "medianImpute")
train_mi <- x_train_mi %>% mutate(SalePrice = train_data[, ncol(train_data)])
test_mi <- x_test_mi %>% mutate(SalePrice = test_data[, ncol(test_data)])
glimpse(train_mi)
glimpse(test_mi)
nrow(test_mi) + nrow(train_mi)


### Data preprocessing - Median imputation, Centering and Scaling 
x_train_mcs <- pre_process(x_train, c("medianImpute", "center", "scale"))
x_test_mcs <- pre_process(x_test, c("medianImpute", "center", "scale"))
train_mcs <- x_train_mcs %>% mutate(SalePrice = train_data[, ncol(train_data)])
test_mcs <- x_test_mcs %>% mutate(SalePrice = test_data[, ncol(test_data)])
glimpse(train_mcs)
glimpse(test_mcs)
nrow(test_mcs) + nrow(train_mcs)


### Data preprocessing - knn
library("RANN")
x_train_knn <- pre_process(x_train,"knnImpute")
x_test_knn <- pre_process(x_test, "knnImpute")
train_knn <- x_train_knn %>% mutate(SalePrice = train_data[, ncol(train_data)])
test_knn <- x_test_knn %>% mutate(SalePrice = test_data[, ncol(test_data)])
glimpse(train_knn)
glimpse(test_knn)
nrow(test_knn) + nrow(train_knn)


## Modelling
### Train Control with 5X5 folds cross validation
myControl <- trainControl(method="repeatedcv", number = 5, repeats = 5, verboseIter = TRUE)

### Linear Model
#### Linear Model with data preprocessed using Median imputation 
model_lm_mi <- train( SalePrice ~ ., data = train_mi, method="lm",
                      trControl = myControl)
model_lm_mi

#### Linear Model with data preprocessed using Median imputation, Centering and Scaling
model_lm_mcs <- train( SalePrice ~ ., data = train_mcs, method="lm",
                       trControl = myControl)
model_lm_mcs
#### Linear Model with data preprocessed using knn
model_lm_knn <- train( SalePrice ~ ., data = train_knn, method="lm",
                       trControl = myControl)
model_lm_knn
#### Compare the three models using the RMSE
lm_list <- list(lm_mi = model_lm_mi, lm_mcs = model_lm_mcs, lm_knn = model_lm_knn)
resamples <- resamples(lm_list)
summary(resamples)
bwplot(resamples, metric="RMSE")


### Random forest
#### Random Forest with data preprocessed using Median imputation 
model_rf_mi <- train( SalePrice ~ ., data = train_mi, method="ranger",
                      trControl = myControl)
model_rf_mi
#### Random Forest with data preprocessed using Median imputation, Centering and Scaling 
model_rf_mcs <- train( SalePrice ~ ., data = train_mcs, method="ranger",
                       trControl = myControl)
model_rf_mcs
#### Random Forest with data preprocessed using knn
model_rf_knn <- train( SalePrice ~ ., data = train_knn, method="ranger",
                       trControl = myControl)
model_rf_knn
#### Compare the three models using the RMSE
rf_list <- list(rf_mi = model_rf_mi, rf_mcs = model_rf_mcs, rf_knn = model_rf_knn)
resamples <- resamples(rf_list)
summary(resamples)
bwplot(resamples, metric="RMSE")
## Compare all the models

all_list <- append(lm_list, rf_list)
all_list
resamples <- resamples(all_list)
summary(resamples)
bwplot(resamples, metric="RMSE")


#### Random Forest with data preprocessed using Median imputation, Centering and Scaling
pred_df <- evalute_model(model_rf_mcs, test_mcs, test_mcs$SalePrice)
head(pred_df)
defaultSummary(pred_df)
xyplot(pred_df$obs ~ pred_df$pred, type = c("p", "g"), xlab = "Predicted", ylab = "Observed")



#######BORUTA
library(data.table)
library(Boruta)
library(plyr)
library(dplyr)
library(pROC)

ID.VAR <- "Id"
TARGET.VAR <- "SalePrice"

sample.df <- read.csv("train.csv",stringsAsFactors = FALSE)

candidate.features <- setdiff(names(sample.df),c(ID.VAR,TARGET.VAR))
data.type <- sapply(candidate.features,function(x){class(sample.df[[x]])})
table(data.type)

explanatory.attributes <- setdiff(names(sample.df),c(ID.VAR,TARGET.VAR))
data.classes <- sapply(explanatory.attributes,function(x){class(sample.df[[x]])})

# categorize data types in the data set?
unique.classes <- unique(data.classes)

attr.data.types <- lapply(unique.classes,function(x){names(data.classes[data.classes==x])})
names(attr.data.types) <- unique.classes


# pull out the response variable
response <- sample.df$SalePrice

lmdata<- sample.df
sample.df <- sample.df[-sample.df$SalePrice]

# remove identifier and response variables
sample.df <- sample.df[candidate.features]

# for numeric set missing values to -1 for purposes of the random forest run
for (x in attr.data.types$integer){
  sample.df[[x]][is.na(sample.df[[x]])] <- -1
}

for (x in attr.data.types$character){
  sample.df[[x]][is.na(sample.df[[x]])] <- "*MISSING*"
}

set.seed(13)
bor.results <- Boruta(sample.df,response,maxRuns=100,doTrace=0)
print(bor.results)
plot(bor.results)

attr <- getSelectedAttributes(bor.results, withTentative = T)
attr

###########################Training/ Testing ########################


ab<- lmdata[,c(attr,TARGET.VAR)]

noofrows <- nrow(ab)
set.seed(12345)
index <- sample(noofrows, 0.6*noofrows, replace=FALSE)
train<- ab[index,]
train

validate <- ab[-index,]
validate


## Data Exploration
glimpse(ab)
summary(ab)
sum(is.na(ab))

## Data cleansing
sum(is.na(ab$Alley))
ab$Alley <- NULL

sum(is.na(ab$PoolQC))
ab$PoolQC <- NULL

sum(is.na(ab$Fence))
ab$Fence <- NULL

sum(is.na(ab$MiscFeature))
ab$MiscFeature <- NULL

ab$Id<- NULL

ab_clean <- change_char_to_factor(ab)

glimpse(ab_clean)
sum(is.na(ab))

## Split the data into training and testing datasets
set.seed(42)
rows <- sample(nrow((ab_clean)))
train <- ab_clean[rows,]
split <- round(nrow(ab_clean)*.80)
ab_data <- train[1:split, ]
test_data <- train[(split + 1):nrow(train), ]
nrow(ab_data) + nrow(ab_data)

## Data preprocessing
x_train <- ab_data[-ncol(ab_data)]
y_train <- ab_data %>% select(SalePrice)
x_test <- ab_data[-ncol(ab_data)]
y_test <- ab_data %>% select(SalePrice)

### Data preprocessing - Median imputation 
x_train_mi <- pre_process(x_train, "medianImpute")
x_test_mi <- pre_process(x_test, "medianImpute")
train_mi <- x_train_mi %>% mutate(SalePrice = ab_data[, ncol(ab_data)])
test_mi <- x_test_mi %>% mutate(SalePrice = ab_data[, ncol(ab_data)])
glimpse(train_mi)
glimpse(test_mi)
nrow(test_mi) + nrow(train_mi)

### Data preprocessing - Median imputation, Centering and Scaling 
x_train_mcs <- pre_process(x_train, c("medianImpute", "center", "scale"))
x_test_mcs <- pre_process(x_test, c("medianImpute", "center", "scale"))
train_mcs <- x_train_mcs %>% mutate(SalePrice = ab_data[, ncol(ab_data)])
test_mcs <- x_test_mcs %>% mutate(SalePrice = ab_data[, ncol(ab_data)])
glimpse(train_mcs)
glimpse(test_mcs)
nrow(test_mcs) + nrow(train_mcs)

### Data preprocessing - knn
x_train_knn <- pre_process(x_train,"knnImpute")
x_test_knn <- pre_process(x_test, "knnImpute")
train_knn <- x_train_knn %>% mutate(SalePrice = ab_data[, ncol(ab_data)])
test_knn <- x_test_knn %>% mutate(SalePrice = ab_data[, ncol(ab_data)])
glimpse(train_knn)
glimpse(test_knn)
nrow(test_knn) + nrow(train_knn)

## Modelling
myControl <- trainControl(method="repeatedcv", number = 5, repeats = 5, verboseIter = TRUE)

### Linear Model
#### Linear Model with data preprocessed using Median imputation 
model_lm_mi <- train( SalePrice ~ ., data = train_mi, method="lm",trControl = myControl)
model_lm_mi
#### Linear Model with data preprocessed using Median imputation, Centering and Scaling
model_lm_mcs <- train( SalePrice ~ ., data = train_mcs, method="lm",trControl = myControl)
model_lm_mcs
#### Linear Model with data preprocessed using knn
model_lm_knn <- train( SalePrice ~ ., data = train_knn, method="lm",trControl = myControl)
model_lm_knn
#### Compare the three models using the RMSE
lm_list <- list(lm_mi = model_lm_mi, lm_mcs = model_lm_mcs, lm_knn = model_lm_knn)
resamples <- resamples(lm_list)
summary(resamples)
bwplot(resamples, metric="RMSE")



### Random forest
#### Random Forest with data preprocessed using Median imputation 
model_rf_mi <- train( SalePrice ~ ., data = train_mi, method="ranger",trControl = myControl)
model_rf_mi
#### Random Forest with data preprocessed using Median imputation, Centering and Scaling 
model_rf_mcs <- train( SalePrice ~ ., data = train_mcs, method="ranger",trControl = myControl)
model_rf_mcs
#### Random Forest with data preprocessed using knn
model_rf_knn <- train( SalePrice ~ ., data = train_knn, method="ranger",trControl = myControl)
model_rf_knn
#### Compare the three models using the RMSE
rf_list <- list(rf_mi = model_rf_mi, rf_mcs = model_rf_mcs, rf_knn = model_rf_knn)
resamples <- resamples(rf_list)
summary(resamples)
bwplot(resamples, metric="RMSE")

## Compare all the models
all_list <- append(lm_list, rf_list)
all_list
resamples <- resamples(all_list)
summary(resamples)
bwplot(resamples, metric="RMSE")

## Evaluate the Models on test data
### Random Forest
#### Random Forest with data preprocessed using Median imputation, Centering and Scaling
pred_df <- evalute_model(model_rf_mcs, test_mcs, test_mcs$SalePrice)
head(pred_df)
defaultSummary(pred_df)
xyplot(pred_df$obs ~ pred_df$pred, type = c("p", "g"), xlab = "Predicted", ylab = "Observed")

















