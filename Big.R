# Importation des packages 

library(gbm)
library(dplyr)
library(caret)

library(randomForest)
library(plotly)

library(ROCR)
library(pROC)

library(glmnet)
data=read.csv("C:/Master/M2/Projet Big DATA/dossierBigDataAnalytics (2)/dossierBigDataAnalytics/data.csv",header = TRUE, sep= ";", encoding="UTF-8") 
brute =read.csv("C:/Master/M2/Projet Big DATA/dossierBigDataAnalytics (2)/dossierBigDataAnalytics/monfichier.csv",header = TRUE, sep= ";", encoding="UTF-8") 

train=train[1:100,]
valid=valid[1:40,]
# Importation de la base de données 
brute =read.table("C:/Master/M2/Projet Big DATA/dossierBigDataAnalytics (2)/dossierBigDataAnalytics/kaggle.txt") 

dim(data)
summary(data)


# On aurait pu les remplacer par des valeurs médianes avec na.action = na.roughfix dans les algo

# Renommer les colonnes 
# On supprime la première colonne qui n'est rien d'autres que l'index 

brute=brute[,-1]

colnames(brute)[1] <- 'SeriousDlqin2yrs' 
colnames(brute)[2] <- 'RevolvingUtilizationOfUnsecuredLines'  
colnames(brute)[3] <- 'Age' 
colnames(brute)[4] <- 'NumberOfTime30-59DaysPastDueNotWorse' 
colnames(brute)[5] <- 'DebtRatio ' 
colnames(brute)[6] <- 'MonthlyIncome'  
colnames(brute)[7] <- 'NumberOfOpenCreditLinesAndLoans' 
colnames(brute)[8] <- 'NumberOfTimes90DaysLate' 
colnames(brute)[9] <- 'NumberRealEstateLoansOrLines'  
colnames(brute)[10] <- 'NumberOfTimes60-89DaysPastDueNotWorse' 
colnames(brute)[11] <- 'NumberOfDependents' 


brute2=read.csv("C:/Master/M2/Projet Big DATA/dossierBigDataAnalytics (2)/dossierBigDataAnalytics/monfichier.csv",header = TRUE, sep= ";", encoding="UTF-8") 
# Traitement des valeurs manquantes
is.na(brute)
data=na.omit(brute)

# Partition
# Séparation de la table en échantillon d'apprentissage (70%) et échantillon test (30%)

S=0.7 
perm = sample(1:nrow(data),ceiling(nrow(data)*S))
train = data[perm,] 
valid = data[-perm,]
attach(train)


###################      RANDOM FOREST      #####################

#Implantation du Random Forest
set.seed(123)

model_RF <- randomForest(train$SeriousDlqin2yrs~.,data=train)
print(model_RF)

plot(model_RF ,col="blue",cex=4)
summary(model_RF)

# Importance des variables
varImpPlot(model_RF)

# Récupération des predictions

pred_RF =predict(model_RF,newdata=valid,type="response") 
print(pred_RF[1:5])
confusionMatrix(pred_RF, as.factor(valid$SeriousDlqin2yrs))

model_RF_test <- randomForest(train$SeriousDlqin2yrs~.,data=train, ntree = 1000, 
                    mtry = 3)
print(model_RF_test)
plot(model_RF_test$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")
# On remarque que l'érreur se stabilise à partir de 200 arbres 
# Donc le nombre optimal est de 200

model_RF_final<- randomForest(train$SeriousDlqin2yrs~.,data=train, ntree = 200, 
                    mtry = 3)
print(model_RF_final)

pred_fit =predict(model_RF_final,newdata=valid,type="response") 
confusionMatrix(pred_fit, as.factor(valid$SeriousDlqin2yrs))
print(summary(pred_fit))
typeof(pred_fit)

# Courbe ROC 
roc_RF=prediction(as.numeric(pred_fit),valid$SeriousDlqin2yrs)
perf_RF=performance(roc_RF,"tpr", "fpr")
plot(perf_RF)

rr=prediction(as.numeric(pred_fit),valid$SeriousDlqin2yrs)
sensispe=performance(rr,"sens", "spec")

# AUC
roc1= roc(valid$SeriousDlqin2yrs,as.numeric(pred_fit),plot=TRUE,legacy.axes=TRUE, lwd=2, col="blue",print.auc=TRUE,grid=TRUE)
auc=as.vector(roc1$auc)
# On a une AUC de 57%
gini=2*auc -1
# et un gini de 15%

# Taux d'erreur

RF.pred=rep(0,nrow(valid))
RF.pred[as.numeric(pred_fit)>0.5]=1
tx_err=as.vector(mean(RF.pred!=valid$SeriousDlqin2yrs))


# Autres
ks=as.vector(ks.test(unlist(sensispe@x.values),unlist(sensispe@y.values))$statistic)
gini=2*auc -1
final_auc=c(auc,tx_err,gini,ks)
final_auc


###################      REGRESSION LOGISTIQUE     #####################


model_reg=glm(SeriousDlqin2yrs~., family=binomial(link=logit), data=train)
summary(model_reg)

valid.p <- cbind(valid, predict(model_reg, newdata = valid, type = "link", 
                                  se = TRUE))
head(valid.p)

valid.p <- within(valid.p, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})
tail(valid.p)

valid.p <- cbind(valid.p, pred.cible = factor(ifelse(valid.p$PredictedProb > 
                                                       0.5, 1, 0)))
head(valid.p)

# Matrice de confusion
m.confusion <- as.matrix(table(valid.p$pred.cible, valid.p$SeriousDlqin2yrs))

# Il faut la ploter
m.confusion <- unclass(m.confusion)


# Taux d'erreur

Tx_err <- function(y, ypred) {
  mc <- table(y, ypred)
  error <- (mc[1, 2] + mc[2, 1])/sum(mc)
  print(error)
}
Tx_err(valid.p$pred.cible, valid.p$SeriousDlqin2yrs)

# COURBE ROC
Pred = prediction(valid.p$PredictedProb, valid.p$SeriousDlqin2yrs)
Perf = performance(Pred, "tpr", "fpr")
plot(Perf, colorize = TRUE, main = "Courbe ROC")

# AUC 
perf <- performance(Pred, "auc")
perf@y.values[[1]]


###################      LASSO     #####################

# The glmnet function does not work with dataframes
# so we need to create a numeric matrix for the training features and a vector of target values.

# Dumy code categorical predictor variables
x <- model.matrix(SeriousDlqin2yrs~., train)[,-1]
# Convert the outcome (class) to a numerical variable
y <- ifelse(train$SeriousDlqin2yrs == "pos", 1, 0)

#
# alpha: the elasticnet mixing parameter. Allowed values include:
#  “1”: for lasso regression
#“0”: for ridge regression
# a value between 0 and 1 (say 0.3) for elastic net regression.

print()
