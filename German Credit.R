#German credit data
german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

dim(german_credit)
head(german_credit)
str(german_credit)
summary(german_credit)
names(german_credit)
table(german_credit$response)

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", 
                            "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install",
                            "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

par(mfrow=c(1,1))
hist(german_credit$duration, xlab = "Duration", main = "Distribution for Duration")
hist(german_credit$amount, xlab = "Amount", main = "Distribution for Amount")
hist(german_credit$installment_rate, xlab = "Installment Rate", main = "Distribution for Installment Rate")
hist(german_credit$present_resid, xlab = "present_resid", main = "Distribution for present_resid")
hist(german_credit$age, xlab = "age", main = "Distribution for Age")
hist(german_credit$n_credits, xlab = "Credits", main = "n_credits")
hist(german_credit$n_people, xlab = "n_people", main = "Distribution for n_people")
hist(german_credit$response, xlab = "response", main = "Distribution for response")
###############
design=german_credit[,-c(1,3,4,6,7,9,10,12,14,15,17,19,20,21)]
cormat=cor(design)
library(corrplot)
corrplot(cormat)

###############
# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$response = german_credit$response - 1
###


glm0=glm(response~., family=binomial, data=german_credit)
german_pred<-predict(glm0,type="response")

dim(german_credit)
colnames(german_credit)

set.seed(12383328)
subset <- sample(nrow(german_credit), nrow(german_credit) * 0.8)
credit.train = german_credit[subset, ]
credit.test = german_credit[-subset, ]

credit.glm0 <- glm(response ~ . , family = binomial, data = credit.train)
credit_null <- glm(response ~ 1, family = binomial, data = credit.train )

summary(credit.glm0)

#deviance residuals:
#https://www.rdocumentation.org/packages/binomTools/versions/1.0-1/topics/Residuals

#Stepwise varaible selection
credit.glm.step <- step(credit.glm0,
                        scope = list(lower= credit_null, 
                                     upper = credit.glm0),
                        direction = "both")

AIC(credit.glm.step)

summary(credit.glm.step)
###################################

library(car)

vif(credit.glm0)





###################################



credit_model1 <- glm(response ~ chk_acct + duration + credit_his + 
                       purpose + amount + 
                       saving_acct + present_emp + installment_rate +
                       sex + other_debtor + 
                       age + other_install + foreign,
                     family = binomial, credit.train)
summary(credit_model1)
AIC(credit_model1)
hist(predict(credit_model1, type = "response"), 
     xlab = "predicted response probability", 
     main = "Predicted response for AIC Model")  

probs_1=predict(credit_model1, type = "response")
summary(probs_1)

############# symmetric cost function
searchgrid = seq(0.01, 0.99, 0.01)

cost0 <- function(r, pi) {
  mean(((r == 0) & (pi > pcut)) | ((r == 1) & (pi < pcut)))
}

result = cbind(searchgrid, NA)

for (i in 1:length(searchgrid)) {
  pcut <- result[i, 1]
  # assign the cost to the 2nd col
  result[i, 2] <- cost0(credit.train$response, predict(credit.glm0, type = "response"))
}

opt_cut=searchgrid[which(result[,2]==min(result[,2]))]

cut = 0.42

prob.glm1.insample <- predict(credit.glm0, type = "response")
predicted.glm1.insample <- prob.glm1.insample > cut
predicted.glm1.insample <- as.numeric(predicted.glm1.insample)


#to create confusion matrix
table(credit.train$response, predicted.glm1.insample, dnn = c("Truth", "Predicted"))

##table(predict(credit_model1, type = "response") > 0.5)
##table(predict(credit_model1, type = "response") > 0.2)
#In-sample and out-of-sample prediction
cutoff= 0.55

prob.glm1.insample <- predict(credit_model1, type = "response")
predicted.glm1.insample <- prob.glm1.insample > cutoff
predicted.glm1.insample <- as.numeric(predicted.glm1.insample)


#to create confusion matrix
table(credit.train$response, predicted.glm1.insample, dnn = c("Truth", "Predicted"))


##install.packages("verification")
library("verification")

cost <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
cost(credit.train$response, predicted.glm1.insample)
#to compute misclassification rate
mean(ifelse(credit.train$response != predicted.glm1.insample, 1, 0))

roc_curve=roc.plot(x=(credit.train$response == "1"), pred =prob.glm1.insample)
roc_curve$roc.vol

## Out-of-sample (performance on testing set)
prob.glm1.outsample <- predict(credit_model1, credit.test, type = "response")
predicted.glm1.outsample <- prob.glm1.outsample > cutoff
predicted.glm1.outsample <- as.numeric(predicted.glm1.outsample)
table(credit.test$Y, predicted.glm1.outsample, dnn = c("Truth", "Predicted"))

cost(credit.test$response, predicted.glm1.outsample)

mean(ifelse(credit.test$response != predicted.glm1.outsample, 1, 0))
roc_curve=roc.plot(x=(credit.test$response == "1"), pred =prob.glm1.outsample)
roc_curve$roc.vol

##############################CART
library(rpart)
credit.rpart <- rpart(formula = response ~ . , data = credit.train, method = "class", 
                      parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)))
plot(credit.rpart)
text(credit.rpart)
names(credit.rpart)

credit.train.a = predict(credit.rpart, credit.train, type = "class")
table(credit.train$response, credit.train.a, dnn = c("Truth", "Predicted"))

credit.test.b = predict(credit.rpart, credit.test, type = "class")
table(credit.test$response, credit.test.b, dnn = c("Truth", "Predicted"))


cost <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
cost(credit.train$response, credit.train.a)
mean(ifelse(credit.test$response != credit.test.b, 1, 0))
credit.rpart2 <- rpart(formula = response ~ ., data = credit.train, method = "class", 
                       cp = 5e-04)

mean(ifelse(credit.test$response != credit.train.a, 1, 0))
# Probability of getting 1
#credit.test.prob.rpart2 = predict(credit.rpart2, credit.test, type = "prob")
install.packages("ROCR")
library(ROCR)
###ROC
credit.train.prob.rpart2 = predict(credit.rpart2, credit.train, type = "prob")
iu <- credit.train$response
pred = prediction(credit.train.prob.rpart2[, 2], credit.train$response)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]

###Out of sample

cost(credit.test$response, credit.test.b)

credit.test.prob.rpart2 = predict(credit.rpart2, credit.test, type = "prob")
pred1 = prediction(credit.test.prob.rpart2[, 2], credit.test$response)
perf1 = performance(pred1, "tpr", "fpr")
plot(perf1, colorize = TRUE)

slot(performance(pred1, "auc"), "y.values")[[1]]

###############################GAM

library(mgcv)
boston.gam <- gam(medv~s(crim)+zn+s(indus)+chas+s(nox)+s(rm)+age+
                    s(dis)+rad+s(tax)+s(ptratio)+
                    black+s(lstat), data=Boston.train)

credit_model1 <- gam(response ~ chk_acct + s(duration) + credit_his + 
                       purpose + s(amount) + 
                       saving_acct + present_emp + s(installment_rate) +
                       s(present_resid)+ 
                       sex + other_debtor+
                       s(age) + other_install + foreign,
                     family = binomial, credit.train)

gam_formula <- as.formula(paste("Y~s(duration)+s(amount) +s(installment_rate)
                                + s(present_resid) + s(age) + s(n_credits) + 
                                s(n_people)+ chk_acct + credit_his +
                                purpose + saving_acct + present_emp+
                                sex + other_debtor+ property + other_install+
                                housing + job + telephone+
                                foreign +", 
                                paste(colnames(credit.train)[6:61], 
                                                                   
                                      collapse = "+")))

credit.gam <- gam(response~s(duration)+s(amount)+s(installment_rate)
                  + s(present_resid) + s(age) + s(n_credits) + 
                    s(n_people)+ chk_acct + credit_his +
                    purpose + saving_acct + present_emp+
                    sex + other_debtor+ property + other_install+
                    housing + job + telephone+
                    foreign , family = binomial, data = credit.train)

credit.gam <- gam(response~duration+s(amount)+age + installment_rate
                  + present_resid  + n_credits + 
                    n_people+ chk_acct + credit_his +
                    purpose + saving_acct + present_emp+
                    sex + other_debtor+ property + other_install+
                    housing + job + telephone+
                    foreign
                   , family = binomial, data = credit.train)



summary(credit.gam)
plot(credit.gam, shade = TRUE, scale = 0)

pcut.gam <- 1/6
prob.gam.in <- predict(credit.gam, credit.train, type = "response")
pred.gam.in <- (prob.gam.in >= pcut.gam) * 1
table(credit.train$response, pred.gam.in, dnn = c("Observation", "Prediction"))

mean(ifelse(credit.train$response != pred.gam.in, 1, 0))
credit.gam$deviance

roc_curve1=roc.plot(x=(credit.train$response == "1"), pred =prob.gam.in)
roc_curve1$roc.vol

prob.gam.out <- predict(credit.gam, credit.test, type = "response")
pred.gam.out <- (prob.gam.out >= pcut.gam) * 1
table(credit.test$response, pred.gam.out, dnn = c("Observation", "Prediction"))

mean(ifelse(credit.test$response != pred.gam.out, 1, 0))

roc_curve1=roc.plot(x=(credit.test$response == "1"), pred =prob.gam.out)
roc_curve1$roc.vol

####################Neural net


#####################Discriminant Analysis

# In-sample
credit.train$response = as.factor(credit.train$response)
credit.lda <- lda(response ~ ., data = credit.train)
summary(credit.lda)

#credit.lda$deviance

prob.lda.in <- predict(credit.lda, data = credit.train)
pcut.lda <- 1/6  
pred.lda.in <- (prob.lda.in$posterior[, 2] >= pcut.lda) * 1
table(credit.train$response, pred.lda.in, dnn = c("Obs", "Pred"))
creditcost(credit.train$Y, pred.lda.in)
mean(ifelse(credit.train$response != pred.lda.in, 1, 0))

roc_curve=roc.plot(x=(credit.train$response == "1"), pred =prob.lda.in$posterior[, 2])
roc_curve$roc.vol

#out-of-sample
lda.out <- predict(credit.lda, newdata = credit.test)
cut.lda <- pcut.lda
pred.lda.out <- as.numeric((lda.out$posterior[, 2] >=cut.lda))
table(credit.test$response, pred.lda.out, dnn = c("Obs", "Pred"))

mean(ifelse(credit.test$response != pred.lda.out, 1, 0))
creditcost(credit.test$Y, pred.lda.out)
predicted.glm1.outsample <- as.numeric(predicted.glm1.outsample)
roc_curve1=roc.plot(x=(credit.test$response == "1"), pred =lda.out$posterior[, 2])
roc_curve1$roc.vol























