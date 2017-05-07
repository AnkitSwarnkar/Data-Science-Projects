setwd("C:\\Users\\ankitswarnkar\\Projects\\Data Mining Project")
library(readxl)
library(MASS)
library(ROSE) # imbalance data
library(readxl)
library(glmnet)
default_of_credit_card_clients <- read_excel("default of credit card clients.xls", 
                                             col_types = c("numeric", "numeric", "numeric", 
                                                           "numeric", "numeric", "numeric", "numeric", "numeric", 
                                                           "numeric", "numeric","numeric", "numeric", "numeric", 
                                                           "numeric", "numeric", "numeric", "numeric","numeric", 
                                                           "numeric", "numeric", "numeric", "numeric", "numeric", 
                                                           "numeric", "numeric"))
View(default_of_credit_card_clients)
data=default_of_credit_card_clients
data = data[-1,-1]
data <- ovun.sample(Y ~ ., data = data, method = "over", N = 50000)$data
data_ccc = data
smp_size <- floor(0.7 * nrow(data_ccc))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(data_ccc)), size = smp_size)

train_ccc <- data_ccc[train_ind,1:23]
test_ccc <- data_ccc[-train_ind, ]
test_ccc_feature = test_ccc[,1:23] 
test_ccc_class = test_ccc[,24]
Y = data_ccc[train_ind,24]
X = cbind(train_ccc,Y)

### Logistic Regression ##
x_train = as.matrix(train_ccc)
y_train = as.matrix(Y)
logit.model<-glmnet(x=x_train, y=y_train,alpha=1,standardize=FALSE, family="binomial")
cv.glmmod <- cv.glmnet(x=x_train, y=y_train, alpha=1)
plot(cv.glmmod)
best.lambda <- cv.glmmod$lambda.min

#Prediction#
x_test = as.matrix(test_ccc_feature)
logit.result <- predict(object = logit.model,s=0.01,x_test,type="response")
logit.result <- ifelse(logit.result > 0.55,1,0)
misClasificError <- sum(logit.result!=test_ccc_class)/nrow(logit.result)
accuracy_lr = 1 - misClasificError

print(paste('Accuracy LR',1-misClasificError))

# Model 2 : SVM ###
#install.packages("e1071")
library(e1071)
svm1 <- svm(Y ~ ., data=X)
# Predictions
predicted_svm = predict(svm1,test_ccc_feature)
accuracy_svm = sum(test_ccc_class == round(predicted_svm)) / length(test_ccc_class)
print(paste('Accuracy for svm',accuracy_svm))



# Random Forest prediction of seeds data
#install.packages("randomForest")
library(randomForest)
fit <- randomForest(Y ~ .-Y, data=X, ntree=1000,do.trace=T)
print(fit) # view results 
importance(fit) # importance of each predictor
predicted_rf = predict(fit,test_ccc_feature)
accuracy_rf = sum(test_ccc_class == round(predicted_rf)) / length(test_ccc_class)
print(paste('Accuracy for Random Forest',accuracy_rf))

results <- sample(list(logistic=logit.result, svm=svm1,rf=fit))
# Table comparison
summary(results)

classifier_name = c("Logistic Regression" = accuracy_lr ,"SVM" = accuracy_svm,"Random Forests" = accuracy_lr)