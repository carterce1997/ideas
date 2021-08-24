library(xgboost)
library(magrittr)
library(Matrix)
library(caret)
library(doSNOW)
library(pROC)
library(mlbench)
library(vroom)
library(tidyverse)


#Read in data
df1 = vroom(file = "~/Desktop/heathcaredata.csv" )

#Select certain columns from the original df 
patient_med = df1 %>%  select(3:5,42,49,50)

#Replace ? character w/ NA
patient_med[ patient_med == "?" ] <- NA

#Add a feature for missing Race, don't want to delete data as it can be 
#indicative of a patter if that data was present. 
patient_med$MissingRace<-ifelse(is.na(patient_med$race), "Y","N")

#Viewing table contents of revised df 
table(patient_med$race)

#Converting cat to numeric values
Age <- c("[0-10)"=0,"[10-20)"=1,"[20-30)"=2, "[30-40)"=3, "[40-50)"=4, "[50-60)"=5,"[60-70)"=6, "[70-80)"=7,  "[80-90)"=8,"[90-100)"=9 )
patient_med$age <- Age[patient_med$age]

Readmitted <- c(">30" = 1, "NO" = 0, "<30" = 2)
patient_med$readmitted <- Readmitted[patient_med$readmitted]

Diabetes <- c("Yes" = 1, "No" = 2)
patient_med$diabetesMed <- Diabetes[patient_med$diabetesMed]

Insulin <- c("Up" = 1, "Down" = 3, "Steady" = 4, "No"=2)
patient_med$insulin <- Insulin[patient_med$insulin]

Gender <- c("Male" = 1, "Female" = 2)
patient_med$gender <- Gender[patient_med$gender]

Race<-c("Caucasian"=1,"AfricanAmerican"=2,"Hispanic"=3,"Asian"=4,"Other"=5)
patient_med$race <- Race[patient_med$race]



#convert  to factor 
patient_med$race=as.factor(patient_med$race)
patient_med$gender=as.factor(patient_med$gender)
patient_med$age=as.factor(patient_med$age)
patient_med$insulin=as.factor(patient_med$insulin)
patient_med$diabetesMed=as.factor(patient_med$diabetesMed)
patient_med$readmitted=as.factor(patient_med$readmitted)
patient_med$MissingRace=as.factor(patient_med$MissingRace)


str(patient_med)

patient_med = patient_med[!is.na(patient_med$race),]


#Rearrange cols
patient_med <- patient_med[, c(5, 1, 2, 3,4,6,7)]

#Must transform vars to dummy vars 
dummy.vars <- dummyVars(~.,data = patient_med[, -1])
patient_med.dummy <- predict(dummy.vars,patient_med[, -1])
View(patient_med.dummy)


#Impute, only doing this bc dataset is 10k obs 
pre.process <- preProcess(patient_med.dummy,method = "bagImpute")
imputed.data <-predict(pre.process,patient_med.dummy)
View(imputed.data)

patient_med$MissingRace <-imputed.data[25,26]

#Splitting Data 
set.seed(54321)
indexes <- createDataPartition(patient_med$diabetesMed, 
                               times = 1,
                               p = 0.7,
                               list = FALSE)

patient_med.train <-patient_med[indexes,-7]
patient_med.test  <-patient_med[indexes,-7]

#Examine proportions, trying to determine potential imbalances 
prop.table(table(patient_med$diabetesMed))
prop.table(table(patient_med.train$diabetesMed))
prop.table(table(patient_med.train$diabetesMed))


#Setting up caret do conduct 10x fold cross validation, repeated 3x & to use 
#grid search for optimal hyperparameter values 
train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3, 
                              search = "grid")

#Leverage grid search for hyperparamters for xgboost

tune.grid <- expand.grid(eta = 0.05,
                         nrounds = 10, 
                         max_depth = 8,
                         min_child_weight = 2.0,
                         colsample_bytree = 0.4,
                         gamma = 0,
                         subsample = 1)

#Train model in 10x fold cv repeated 3x & hyperparamter grid search to find opt model 

caret.cv <- train(diabetesMed ~ .,
                  data = na.exclude(patient_med.train),
                  method = "xgbTree", 
                  tuneGrid = tune.grid,
                  trControl = train.control)

#Examine Results 
caret.cv

#Make predictions on the test set trained on all 6840 rows, hyperparamter vals found 
preds <- predict(caret.cv, patient_med.test)

#Use unseen data 30% using confusionMatrix() to determine effectiveness of model 
confusionMatrix(preds,patient_med.test$diabetesMed)


