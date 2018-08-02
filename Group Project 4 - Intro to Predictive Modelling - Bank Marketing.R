## Objective : To maximize the subscription rate for term deposits for a portugese bank
## through a tele marketing campaign

#1.Importing data
#2.Exploratory data analysis
#3.Data treatment
#4.Feature selection
#5.Model fitting

#Note : 
#1.Section 2 can be skipped based on the requirement
#2.Iterations are not included in the code below for various sections of the code

# 1.Importing the data
setwd("--------") ## Please enter the directory path

library(dplyr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(caret)
library(glmnet)
library(ROCR)
library(plyr)
library(randomForest)

raw_data <- read.csv("bank-additional-full.csv",header=T,na.strings=c(""))

# 2.Exploratory data analysis

#### i.Checking the distribution of factor variables in the data and plotting the same
is.fact <- sapply(raw_data, is.factor)
factors.raw <- raw_data[, is.fact]
mdata <- melt(factors.raw, id=c("y"))
k <- dplyr :: summarise(group_by(mdata,variable,value,y),count =n())
uniq <- unique(k$variable)

for (i in uniq){
  idx <- match(i,uniq)
  l = list()
  t <- subset(k,k$variable == i)
  assign(paste("k", i, sep = ""), data.frame(t))
  p <- ggplot(t, aes(factor(value), count, fill = y)) +
    geom_bar(stat="identity", position = "dodge") +
    scale_fill_brewer(palette = "Set1") + ggtitle(i)
  assign(paste("p", idx, sep = ""), p) # need to check the syntax
}

grid.arrange(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10, ncol=3)

## 3.Data treatment
# i.Duration = 0 has to be removed to remove customers that are not contacted
df <- subset(raw_data,duration != 0)
# ii.Create bins for age
df$age <- cut(df$age,c(1,20,40,60,100))
# iii. Checking the missing values
sapply(df,function(x) sum(is.na(x)))
#iv.Setting the response variable to 1,0
df[,'y'] <- sapply(df[,'y'],function(x) ifelse(x=='yes',1,0))
df$y = as.factor(df$y)

## 4. Feature selection
#Lasso regression

set.seed(99)
XXca <- model.matrix(y~., data=data.frame(df))[,-1]
term_deposit = df$y
Lasso.Fit = glmnet(x=XXca,y = term_deposit, family = "binomial", alpha=1)

par(mfrow=c(1,2))
plot(Lasso.Fit)

CV.L = cv.glmnet(XXca,as.numeric(term_deposit),alpha=1)
LamL = CV.L$lambda.1se
plot(log(CV.L$lambda),sqrt(CV.L$cvm),main="LASSO CV (k=10)",xlab="log(lambda)",ylab = "RMSE",col=4,type="b",cex.lab=1.2)
abline(v=log(LamL),lty=2,col=2,lwd=2)

coef.L = predict(CV.L,type="coefficients",s=LamL)
View(coef.L)
##Final subsetting for feature selection
# Features that are selected through Lasso regression
features = c("age.60.100.","jobblue.collar","jobstudent","monthmar","monthmay","monthjul",
             "monthjun","monthnov","day_of_weekmon","poutcomenonexistent",
             "duration","educationuniversity.degree","defaultunknown",
             "pdays","poutcomesuccess","cons.conf.idx","nr.employed","y1")

# Creating a model matrix again for selecting the important features
mod.mat = model.matrix(~., data=data.frame(df))[,-1]
temp_data = data.frame(mod.mat)
temp_data$y1 = as.factor(temp_data$y1)
sel_features_dat_vf <- temp_data[,features] # Dataset used for logistic regression

#5. Model fitting

#########################
## Logistic Regression###
#########################

#### Creating train and test data

set.seed(99)
train <- createDataPartition(y=sel_features_dat_vf$y1,p=0.8,list=FALSE)
train_data <- sel_features_dat_vf[train,]
test_data <- sel_features_dat_vf[-train,]

set.seed(99)
model <- glm(train_data$y1 ~.,family=binomial(link='logit'),data=train_data)
summary(model)

#### Assessing the predictive ability of the model

fitted.results <- predict(model,test_data[,1:17],type='response')
answers = test_data$y1


# Confusion matrix
cm_logistic <- table(answers, fitted.results>= 0.3)

head(fitted.results)

# ROC Curve

ROCRpred <- prediction(fitted.results, test_data$y1)
ROCRperf_log <- performance(ROCRpred, 'tpr','fpr')

p_roc_logistic <- plot(ROCRperf_log, text.adj = c(-0.2,1.7))

#Sensitivity and specificity
sens_spec_ROCR <- performance(ROCRpred, measure = "sens",x.measure = "cutoff")
plot(sens_spec_ROCR, text.adj = c(-0.2,1.7))

# Calculating the cutoff
cost.perf = performance(ROCRpred, "cost")
ROCRpred@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]

# AUC calculation
auc_ROCR <- performance(ROCRpred, measure = "auc")
auc_ROCR

## Cross validation

# False positive rate
fpr <- NULL
# False negative rate
fnr <- NULL
# Number of iterations
k <- 100

# Initialize progress bar
pbar <- create_progress_bar('text')
pbar$init(k)

# Accuracy
acc <- NULL
set.seed(99)

for(i in 1:k)
{
  # Train-test splitting
  # 95% of samples -> fitting
  # 5% of samples -> testing
  smp_size <- floor(0.95 * nrow(sel_features_dat_vf))
  index <- sample(seq_len(nrow(sel_features_dat_vf)),size=smp_size)
  train <- sel_features_dat_vf[index, ]
  test <- sel_features_dat_vf[-index, ]
  
  # Fitting
  model <- glm(y1~.,family=binomial,data=sel_features_dat_vf)
  
  # Predict results
  results_prob <- predict(model,subset(test,select=c(1:17)),type='response')
  
  # If prob > 0.5 then 1, else 0
  results <- ifelse(results_prob > 0.5,1,0)
  
  # Actual answers
  answers <- test$y1
  
  # Accuracy calculation
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
  
  pbar$step()
}

# Average accuracy of the model
mean(acc)

####################
## Random Forests
####################

set.seed(99)
# Inputting all the variables into Random forests
train <- createDataPartition(y=temp_data$y1,p=0.8,list=FALSE)
train_data <- temp_data[train,]
test_data <- temp_data[-train,]

set.seed(99)
# model2 <- randomForest(y1 ~ ., data = train_data, ntree = 20, mtry = 15, importance = TRUE)
model2 <- randomForest(y1 ~ ., data = train_data, ntree = 100, mtry = 15, 
                       na.action=na.fail, importance= TRUE, keep.forest=T)

# Variable importance
varImpPlot(model2)

pred <- predict(model2, test_data, type = "response")
pred.pr <- predict(model2, test_data, type = "prob")[,2]
cm_random_forests <- table(test_data$y1,pred)
cm_random_forests

# Prediction
pred.rf.pred = prediction(pred.pr, test_data$y1)

#performance in terms of true and false positive rates
pred.rf.perf = performance(pred.rf.pred,"tpr","fpr")

#plot the curve
plot(ROCRperf_log, text.adj = c(-0.2,1.7)) # Plot for ROC curve in logistic model
plot(pred.rf.perf,main="ROC Curve for Random Forest",add = TRUE, col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

#compute area under curve
auc <- performance(pred.rf.pred,"auc")
auc

## Conclusion : Logistic Regression gave better results with a threshold of 0.3

