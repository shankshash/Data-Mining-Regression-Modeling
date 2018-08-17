german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

german_credit$response = german_credit$response - 1
#install.packages("dummies")
library(dummies)
set.seed(12383328)
index <- sample(1:nrow(german_credit),round(0.8*nrow(german_credit)))

maxs <- apply(german_credit[,c(2,5,8,11,13,16,18,21)], 2, max) 
mins <- apply(german_credit[,c(2,5,8,11,13,16,18,21)], 2, min)
scaled <- as.data.frame(scale(german_credit[,c(2,5,8,11,13,16,18,21)], center = mins, scale = maxs - mins))
fact <- dummy.data.frame(german_credit[,c(1,3,4,6,7,9,10,12,14,15,17,
                                               19,20)])
comb <- data.frame(scaled,fact)
names(comb)
train <- comb[index,]
test <- comb[-index,]


#install.packages("neuralnet")
library(neuralnet)
Names <- names(train)
f.german <- as.formula(paste("response ~", paste(Names[!Names %in% "response"], collapse = " + ")))

set.seed(12383328)
#Please try different number of hidden layers and different number of neurons in each layer by specifing the sequence in hidden=;
nn1 <- neuralnet(f.german,data=train,hidden= c(18,6),linear.output=F)
plot(nn1)

pr.nn_train <- compute(nn1, comb[,-8])
pr.nn_t <- pr.nn_train$net.result*(max(german_credit$response)-min(german_credit$response))+min(german_credit$response)
train.r <- (train$response)*(max(german_credit$response)-min(german_credit$response))+min(german_credit$response)

MSE.nn.t <- sum((train.r - pr.nn_t)^2)/nrow(comb)
MSE.nn.t

amd <- train.r - pr.nn_t
hist(amd, xlab = "Residuals", main = " Residual Plot", xlim=c(-10,10))
atu <- 1:nrow(amd)    
plot(atu,amd, ylab="residual", xlab="index")
abline(h=0, col = "red")


pr.nn_test <- compute(nn1, test[,1:13])
pr.nn_tt <- pr.nn_test$net.result*(max(german_credit$response)-min(german_credit$response))+min(german_credit$response)
test.r <- (test$response)*(max(german_credit$response)-min(german_credit$response))+min(german_credit$response)

MSE.nn.tt <- sum((test.r - pr.nn_tt)^2)/nrow(test)
MSE.nn.tt

names(nn1)

#prediction
pr.nn <- compute(nn1, test[,1:13])
MSE.nn <- sum((test$response - pr.nn$net.result)^2)/nrow(test)
MSE.nn

#Note: for other options in "neuralnet" package, please also refer to the blog I listed above;


library(nnet)
set.seed(12383328)
credit.nnet <- nnet(response ~ ., data = credit.train, size = 18, decay =0,maxit = 500)

prob.nnet = predict(credit.nnet, credit.train)
pred.nnet1 = as.numeric(prob.nnet > 1/6)
table(credit.train$response, pred.nnet1, dnn = c("Obs.","Prediction"))

mean(ifelse(credit.train$response != pred.nnet1, 1, 0))
roc_curve=roc.plot(x=(credit.train$response == "1"), pred = prob.nnet[,1])
roc_curve$roc.vol

prob.nnet1 = predict(credit.nnet, credit.test)
pred.nnet1 = as.numeric(prob.nnet1 > 1/6)
table(credit.test$response, pred.nnet1, dnn = c("Obs.","Prediction"))

mean(ifelse(credit.test$response != pred.nnet1, 1, 0))
roc_curve=roc.plot(x=(credit.test$response == "1"), pred =prob.nnet1[,1])
roc_curve$roc.vol
#####################Discriminant Analysis

# In-sample
credit.train$response = as.factor(credit.train$response)
credit.lda <- lda(response ~ ., data = credit.train)

prob.lda.in <- predict(credit.lda, data = credit.train)
pcut.lda <- 1/6  
pred.lda.in <- (prob.lda.in$posterior[, 2] >= pcut.lda) * 1
table(credit.train$Y, pred.lda.in, dnn = c("Obs", "Pred"))

creditcost(credit.train$Y, pred.lda.in)
#mean(ifelse(credit.train$Y != pred.lda.in, 1, 0))

#out-of-sample
lda.out <- predict(credit.lda, newdata = credit.test)
cut.lda <- pcut.lda
pred.lda.out <- as.numeric((lda.out$posterior[, 2] >=cut.lda))
table(credit.test$Y, pred.lda.out, dnn = c("Obs", "Pred"))

#mean(ifelse(credit.test$Y != pred.lda.out, 1, 0))
creditcost(credit.test$Y, pred.lda.out)













