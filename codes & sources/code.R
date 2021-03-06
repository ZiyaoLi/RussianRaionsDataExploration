library(stats)
library(MASS)
library(forecast)
library(lars)
data_original=read.csv("moscow_districts.csv")
data=data_original
n=nrow(data);p=ncol(data)


###  first exploration  ###
# pairwise scatter plot
numeric_cols=c(2,3,4,5,6,7,8,18)
data_numeric=data[,numeric_cols]
pairs(data_numeric)
# marginal distr.
par(mfrow=c(2,4))
for (i in 1:8){
  hist(data_numeric[,i], 10,
       main=colnames(data_numeric)[i],
       xlab=colnames(data_numeric)[i])
}
par(mfrow=c(1,1))


###  remove outliers and bad features  ###
outliers=c(27,46,47,61)
bad_features=c(5)
data=data[-outliers,-bad_features]
n=nrow(data);p=ncol(data)


###  marginal transformations  ###
# logit trans. for precentage variables
logit=function(x) {
  y=x+.01
    # considering plausible 0s and percentages all small(<0.8).
  log(y/(1-y))
}
percentage_cols=c("young_all","ekder_all","green_zone_part","indust_part")
for (i in percentage_cols) {
  data[,i]=logit(data[,i])
}
# boxcox trans. for area,popul and avg_houseprice
boxcox_features=c("area_m","popul","avg_houseprice")
boxcox_lambdas=c(0,0,0)
names(boxcox_lambdas)=boxcox_features
for(i in boxcox_features){
  i
  boxcox_lambdas[i]=BoxCox.lambda(data[,i])
  data[,i]=BoxCox(data[,i],boxcox_lambdas[i])
}
# transformed scatter & hist plots
numeric_cols=c(2,3,4,5,6,7,17)
data_numeric=data[,numeric_cols]
pairs(data_numeric)
par(mfrow=c(2,4))
for (i in 1:7){
  hist(data_numeric[,i], 10,
       main=colnames(data_numeric)[i],
       xlab=colnames(data_numeric)[i])
}
par(mfrow=c(1,1))


###  statistical description: multivariate normal? ###
# all variables
x=as.matrix(data[,-1])
z=scale(x)
S=cov(z)
lambda=eigen(S)$values
lambda  # check for condition number: alright (10.21)
chi_x=diag(z%*%solve(S)%*%t(z))
chi_xs=sort(chi_x)
chi_q=qchisq(p=((1:n)-.5)/n,df=ncol(x))
plot(chi_q,chi_xs,main="Chi_Square Plot: Original Data")
lines(x=c(-1,50),y=c(-1,50),col="red")  # can't say normal
# numerical variables
x=as.matrix(data_numeric)
n=nrow(x);p=ncol(x)
z=scale(x)
S=cov(z)
# covariance plot
par(mar=c(7,6,5,4)+.1)
Splot=S[,7:1]
image(Splot,xaxt = 'n', yaxt='n', main="Covariance Plot")
axis(2,labels=colnames(Splot),at=(0:6)/6,las=1,cex.axis=.7)
axis(1,labels=rownames(Splot),at=(0:6)/6,las=3,cex.axis=.7)
par(mar=c(5,4,4,2)+.1)

lambda=eigen(S)$values
lambda  # check for condition number: better (7.11)
chi_x=diag(z%*%solve(S)%*%t(z))
chi_xs=sort(chi_x)
chi_q=qchisq(p=((1:n)-.5)/n,df=ncol(x))
plot(chi_q,chi_xs,main="Chi_Square Plot: Outliers Removed")
lines(x=c(-1,50),y=c(-1,50),col="red")  # can say normal now

###  PCA & Factor Analysis  ###
x=as.matrix(data[,c(-1)])
par(mfrow=c(2,1))
x.pca=princomp(x,cor=T)
cumsdev=cumsum(x.pca$sdev)/sum(x.pca$sdev)
plot(cumsdev,type='b',
     xlab="n_comps",ylab="explained_var",
     main="Explained Variance vs. Components numbers: Correlation")  # no dominant components; not very compressable
x.pca_cov=princomp(x,cor=F)
cumsdev_cov=cumsum(x.pca_cov$sdev)/sum(x.pca_cov$sdev)
plot(cumsdev_cov,type='b',
     xlab="n_comps",ylab="explained_var",
     main="Explained Variance vs. Components numbers: Covariance")  # no dominant components; not very compressable
par(mfrow=c(1,1))
x.fact=factanal(x,5,scores="Bartlett",rotation="varimax")
x.fact$loadings


###  OLS Regression   ###
data_reg=data[,-1]
boxcox_model=lm(avg_houseprice~.,data=data_reg)
summary(boxcox_model)
data_reg[,16]=data_original[-outliers,18]
original_model=lm(avg_houseprice~.,data=data_reg)
summary(original_model)  # R2 of original model is a lot better than a boxcox model
                             # use original house price below.
residuals=original_model$residuals
plot(data_reg[,16],residuals,
     xlab="avg_houseprice",ylab="residuals",
     main="Residual Plot")
lines(x=c(0,2e7),y=c(0,0),col="red",type='c')
abline(lm(residuals~data_reg[,16]),col='blue')

par(mfrow=c(3,5))
for (i in 1:15) {
  plot(data_reg[,i],residuals,
       xlab=colnames(data_reg)[i],ylab="residuals",
       main=colnames(data_reg)[i])
}
par(mfrow=c(1,1))
data_onehot=read.csv("onehot.csv")
onehot_model=lm(avg_houseprice~.,data=data_onehot)
summary(onehot_model)  #  lower R2, but not significant


###  different regression models  ###
rmse=function(true,pred){
  mean((true-pred)^2)^.5
}
train=sample(1:n,size=0.8*n)
test_y=data_reg$avg_houseprice[-train]
ols=lm(avg_houseprice~.,data=data_reg[train,])
ols_pred=predict(ols,data_reg[-train,])
ols_rmse=rmse(test_y,ols_pred)

step_ols=step(ols)
step_ols_pred=predict(step_ols,data_reg[-train,])
step_ols_rmse=rmse(test_y,step_ols_pred)

trainx=as.matrix(data_reg[train,-16])
trainy=data_reg[train,16]
testx=as.matrix(data_reg[-train,-16])

ridge=lm.ridge(avg_houseprice~.,data=data_reg[train,],lambda=seq(0,100,length=10001))
ridge_k=which.min(ridge$GCV)
ridge_coef=coef(ridge)[ridge_k,]
ridge_pred=cbind(1,testx)%*%ridge_coef
ridge_rmse=rmse(test_y,ridge_pred)

lasso=lars(trainx,trainy)
lasso_cv=cv.lars(trainx,trainy,K=5)
lasso_k=lasso_cv$index[which.min(lasso_cv$cv)]
lasso_coef=coef(lasso,s=lasso_k,mode="fraction")
lasso_interc=predict(lasso,s=lasso_k,mode="fraction",newx=t(numeric(15)))$fit
lasso_pred=lasso_interc+testx%*%lasso_coef
lasso_rmse=rmse(test_y,lasso_pred)

totalx=data_reg[,-16]
pcax=predict(princomp(totalx))[,1:10]
totalx=as.data.frame(cbind(pcax,data_reg[,16]))
colnames(totalx)=c(1:10,"hp")
pca=lm(hp~.,data=totalx[train,])
pca_pred=predict(pca,totalx[-train,])
pca_rmse=rmse(test_y,pca_pred)

c(ols_rmse,step_ols_rmse,ridge_rmse,lasso_rmse,pca_rmse)