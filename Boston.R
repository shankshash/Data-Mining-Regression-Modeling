
# DM Lab 2, on Jan 23, 2018
# by Yuankun Zhang


## Part 1 Linear Regression ##
library(MASS)
data(Boston)
dim(Boston) 
names(Boston)
str(Boston)
head(Boston)
summary(Boston)

#check the correlation matrix
design=Boston[,-14]
cormat=cor(design)
cormat

#install.packages("corrplot")
library(corrplot)
corrplot(cormat)

library(lattice)
levelplot(cormat)
#library(fields)
###
?levelplot


summary(Boston)
## Build Models
set.seed(12383328)
index <- sample(nrow(Boston),round(nrow(Boston)*0.80))
Boston.train <- Boston[index,]
Boston.test <- Boston[-index,]
dim(Boston.train)
model0<- lm(medv~lstat, data = Boston.train)
model0
summary(model0)
sum.model0<- summary(model0)
sum.model0
names(sum.model0)
sum.model0$r.squared
sum.model0$adj.r.squared
head(sum.model0$residuals)

#confident intervals for coefficients
confint(model0, level=0.90)

#plot fitted linear line
plot(medv~lstat, data=Boston.train)
abline(model0, col="red", lty=2, lwd=2)


# Prediction on training data set
pred.model0 <- predict(model0, newdata = Boston.train)
head(pred.model0)
predict(model0, newdata = data.frame(lstat=c(5,10,15)), interval = "confidence")



# Multiple linear regreesion
model1<- lm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data=Boston.train)
# more convenient way
model1<- lm(medv~., data=Boston.train)
# summary
summary(model1)

#residual plot
plot(model1$fitted.values, model1$residuals)

par(mfrow=c(1,1)) # to specify the figure layout
plot(model1)


# In-sample performance
sum.model1<- summary(model1)
#MSE
(sum.model1$sigma)^2
sum.model1$r.squared
sum.model1$adj.r.squared
AIC(model1)
BIC(model1)


#reduced model
model2<- lm(medv~. -indus -age, data=Boston.train)
summary(model2)


### Exercise 
pred.model1<- predict(model1,newdata=Boston.train)
mspe.model1<-sum((Boston.train$medv-pred.model1)^2)/nrow(Boston.train) 
mspe.model1 
#check the out-of-sample prediction error 
test.pred.model1<-predict(model1, newdata=Boston.test) 
mpse.model1<-mean((Boston.test$medv-test.pred.model1)^2)
mpse.model1


pred.model2<-predict(model2,newdata=Boston.train)
mspe.model2<-mean((Boston.train$medv-pred.model2)^2)
mspe.model2  
#check the out-of-sample prediction error 
test.pred.model2<-predict(model2,newdata=Boston.test)
length(test.pred.model2)
test.mspe.model2<-mean((Boston.test$medv-test.pred.model2)^2)
test.mspe.model2



## Cross Validation
# library(boot)
# model.glm1 = glm(medv~., data = Boston)
# cv.glm(data = Boston, glmfit = model.glm1, K = 10)$delta[2]



#################################################
## Part 2 Variable Selection ##
#install.packages('leaps')
library(MASS)
library(leaps)
library(glmnet)
library(dplyr)

#data(Boston)
#set.seed(2018)
#index <- sample(nrow(Boston),nrow(Boston)*0.90)
#Boston.train <- Boston[index,]
#Boston.test <- Boston[-index,]

## Linear Regression
model0<- lm(medv~lstat, data = Boston.train)
model1<- lm(medv~., data=Boston.train)
model2<- lm(medv~. -indus -age, data=Boston.train)

#checking model fitting
model0
model1
model2

AIC(model0); BIC(model0)
AIC(model1); BIC(model1)
AIC(model2); BIC(model2)


## Best Subset 
library(leaps)
#regsubsets only takes data frame as input
model.subset<- regsubsets(medv~.,data=Boston.train, nbest=1, nvmax = 13)
model.subset$method
#names(model.subset)

summary(model.subset)
subset_fit=summary(model.subset)
names(subset_fit)

#to get BIC, Cp, R2, adj_R2 etc for each model
cbind(subset_fit$which, subset_fit$bic, subset_fit$rsq, subset_fit$adjr2,subset_fit$cp)
#subset_fit$outmat

par(mfrow=c(1,1))
plot(model.subset, scale="bic")


## Stepwise Selection
## Forward/Backward/Stepwise regression Using AIC (or BIC)
nullmodel<- lm(medv~1, data=Boston.train)
fullmodel<- lm(medv~., data=Boston.train)

#Backward Elimination
model.step.b<- step(fullmodel,direction='backward' )
summary(model.step.b)
#Forward Selection
model.step.f<- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='forward')
summary(model.step.f)

#Stepwise Selection 
model.step.s<- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='both')
summary(model.step.s)

#If you want to conduct Stepwise selection by BIC
n = nrow(Boston.train)
model.step.b<- step(fullmodel,direction='backward', k=log(n))
names(model.step.b)
summary(model.step.b)



############          LASSO         ##################
install.packages("glmnet")
library(glmnet)

#Standardize covariates before fitting LASSO
Boston.X.std<- scale(select(Boston, -medv))
X.train<- as.matrix(Boston.X.std)[index,]
X.test<-  as.matrix(Boston.X.std)[-index,]
Y.train<- Boston[index, "medv"]
Y.test<- Boston[-index, "medv"]

lasso.fit<- glmnet(x=X.train, y=Y.train, family = "gaussian", alpha = 1)
##alpha = 1 performs lassso regression
names(lasso.fit)
plot(lasso.fit, xvar = "lambda", label=TRUE)

#get coefficients fits with different lambdas
coef(lasso.fit, s=0.1)
coef(lasso.fit, s=0.5)
coef(lasso.fit, s=1)
coef(lasso.fit, s=0.02835856)


#CV to select the optimal lambda
cv.lasso<- cv.glmnet(x=X.train, y=Y.train, family = "gaussian", alpha = 1, nfolds = 10)
plot(cv.lasso)
cv.lasso
names(cv.lasso)

#get the optimal lambda that minimizes the mean squared error
cv.lasso$lambda.min
cv.lasso$lambda.1se

pred.lasso.train<- predict(lasso.fit, newx = X.train, s=cv.lasso$lambda.min)
#coef.high<- as.vector(coef(lasso.fit, s=cv.lasso$lambda.1se))
#coef.high[which(coef.high!=0)]


#get prediction on test data set
pred.lasso1<- predict(lasso.fit, newx = X.test, s=0.1)
pred.lasso2<- predict(lasso.fit, newx = X.test, s=0.5)
pred.lasso3<- predict(lasso.fit, newx = X.test, s=1)

pred.lasso.min<- predict(lasso.fit, newx = X.test, s=cv.lasso$lambda.min)
pred.lasso.1se<- predict(lasso.fit, newx = X.test, s=cv.lasso$lambda.1se)

#MSPE
mean((Y.train-pred.lasso.train)^2)
mean((Y.test-pred.lasso.min)^2)

###################
sst <- sum((Y.train - mean(Y.train ))^2)
sse <- sum((pred.lasso.train - Y.train )^2)

# R squared
rsq <- 1 - sse / sst
rsq

sst_1 <- sum((Y.train  - mean(Y.train ))^2)
sse_2 <- sum((pred.lasso.train - Y.train )^2)

# R squared
adjrsq_1 <- 1 - (505*sse_2)/ (492*sst_1)
rsq_1

################################

sst <- sum((Y.test - mean(Y.test))^2)
sse <- sum((pred.lasso.min - Y.test)^2)

# R squared
rsq <- 1 - sse / sst
rsq

sst_1 <- sum((Y.test - mean(Y.test))^2)
sse_2 <- sum((pred.lasso.min - Y.test)^2)

# R squared
adjrsq_1 <- 1 - (101*sse_2)/ (89*sst_1)
rsq_1


## High-dimensional data ##
genedata<- read.csv("http://homepages.uc.edu/~lis6/DataMining/Data/gene_exp.csv")
dim(genedata)

#Regular linear regression won't work
lm.fit.high<- lm(Y~., data = genedata)
lm.fit.high

# Note: we had standardized all covariates before handing out this data set;
lasso.fit.high<- glmnet(x=as.matrix(genedata[,-1]), y=genedata[,1], family = "gaussian", alpha=1)

cv.lasso.fit.high<- cv.glmnet(x= as.matrix(genedata[,-1]), y=genedata[,1], family = "gaussian", alpha=1)
plot(cv.lasso.fit.high)

cv.lasso.fit.high$lambda.min
lasso.best<- glmnet(x=as.matrix(genedata[,-1]), y=genedata[,1], family = "gaussian", alpha=1
                        ,lambda=cv.lasso.fit.high$lambda.min)
coef(lasso.best)
#check the number of selected covariates
sum(coef(lasso.best)!=0)
#get prediction
predict(lasso.best,newx=as.matrix(genedata[,-1]),s=cv.lasso.fit.high$lambda.min)

# alternative way to obtain coefficient estimates
coef.high<- as.vector(coef(lasso.fit.high, s=cv.lasso.fit.high$lambda.min))
coef.high[which(coef.high!=0)]
colnames(genedata[,-1])[which(coef.high!=0)]

?predict
predict


#################################


model1<- lm(medv~crim+zn+chas+nox+rm+ dis+rad+tax+ptratio+black+lstat, data=Boston.train)
summary(model1)
AIC(model1)

pred.model1<-predict(model1,newdata=Boston.train)
mspe.model1<-mean((Boston.train$medv-pred.model1)^2)

pred.model2<-predict(model1,newdata=Boston.test)
mspe.model2<-mean((Boston.test$medv-pred.model2)^2)

################################### CART Model
library(rpart)
boston.rpart <- rpart(formula = medv ~ ., data = Boston.train, cp = 0.013)
boston.rpart

plot(boston.rpart)
text(boston.rpart)

plotcp(boston.rpart)
mean((predict(boston.rpart) - Boston.train$medv)^2)

test.e <- predict(boston.rpart,Boston.test)
mean((test.e- Boston.test$medv)^2)

###################GAM
library(mgcv)
boston.gam <- gam(medv~s(crim)+zn+s(indus)+chas+s(nox)+s(rm)+age+
                    s(dis)+rad+s(tax)+s(ptratio)+
                    black+s(lstat), data=Boston.train)

summary(boston.gam)

b.train.mse <- predict(boston.gam, Boston.train)
mean((b.train.mse- Boston.train$medv)^2)

b.test.mse <- predict(boston.gam, Boston.test)
mean((b.test.mse- Boston.test$medv)^2)

plot(fitted(boston.gam), residuals(boston.gam), xlab='fitted',ylab='residuals',
                                   main='Residuals by fitted for GAM')

######### Neural Network

library(MASS)
data <- Boston

set.seed(12383328)
index <- sample(1:nrow(data),round(0.8*nrow(data)))



#One way to scale your data, but you can use the method in this blog:
#https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/ #
###
#data.nomedv<-data[,-14]

#normalize data with min-max scaling-> (X-Xmin)/(Xmax-Xmin)
#scaling on the training data input variables
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
train <- scaled[index,]
test <- scaled[-index,]

#install.packages("neuralnet")
library(neuralnet)
Names <- names(train)
f.boston <- as.formula(paste("medv ~", paste(Names[!Names %in% "medv"], collapse = " + ")))

set.seed(12383328)
#Please try different number of hidden layers and different number of neurons in each layer by specifing the sequence in hidden=;
nn1 <- neuralnet(f.boston,data=train,hidden= c(5,3),linear.output=T)
  plot(nn1)

pr.nn_train <- compute(nn1, train[,1:13])
pr.nn_t <- pr.nn_train$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
train.r <- (train$medv)*(max(data$medv)-min(data$medv))+min(data$medv)

MSE.nn.t <- sum((train.r - pr.nn_t)^2)/nrow(train)
MSE.nn.t

amd <- train.r - pr.nn_t
hist(amd, xlab = "Residuals", main = " Residual Plot", xlim=c(-10,10))
atu <- 1:nrow(amd)    
plot(atu,amd, ylab="residual", xlab="index")
abline(h=0, col = "red")
     

pr.nn_test <- compute(nn1, test[,1:13])
pr.nn_tt <- pr.nn_test$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test.r <- (test$medv)*(max(data$medv)-min(data$medv))+min(data$medv)

MSE.nn.tt <- sum((test.r - pr.nn_tt)^2)/nrow(test)
MSE.nn.tt

names(nn1)

#prediction
pr.nn <- compute(nn1, test[,1:13])
MSE.nn <- sum((test$medv - pr.nn$net.result)^2)/nrow(test)
MSE.nn

#Note: for other options in "neuralnet" package, please also refer to the blog I listed above;

