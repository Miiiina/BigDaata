#dossier Tokpavi

#arbre de regression & methode de penalisation
#FOCUS sur un probleme de classification: proba de faire defaut
#base de donnée "housing" - 12 variables predictives

# library :
install.packages("mlr")
library(ggplot2)
library(dplyr)
library(shiny)
library(mlr)
library(gbm)
library(dplyr)
library(caret)

library(randomForest)
library(plotly)

library(ROCR)
library(pROC)

library(glmnet)

#roc(test2$reponse,as.vector(glm.probs),plot=TRUE,legacy.axes=TRUE, lwd=2, 
#    col="red",auc.polygon=TRUE,print.auc=TRUE,grid=TRUE)

#================== DATA SET ============================
data=read.csv("C:/Master/M2/Projet Big DATA/dossierBigDataAnalytics (2)/dossierBigDataAnalytics/data.csv",header = TRUE, sep= ";", encoding="UTF-8") 
brute =read.csv("C:/Master/M2/Projet Big DATA/dossierBigDataAnalytics (2)/dossierBigDataAnalytics/monfichier.csv",header = TRUE, sep= ";", encoding="UTF-8") 

#======================== CLEAN THE DATA SET ============================  
clean = as.data.frame(na.omit(brute))

#===================== Transformation des variables nominales en ordinales ===================
clean$cible<- factor(ifelse(clean$SeriousDlqin2yrs == 1, "yes", "no"))

df <- clean

saveRDS(df, file="df.rds")


#======================= matrice de corrélation A RECUPERER =================

#======================= À RECUPERER ===========================
attach(df)
ggplot(data=df, aes(x=df$SeriousDlqin2yrs, fill=SeriousDlqin2yrs)) +
  geom_histogram(binwidth=.5, position="dodge") + 
  labs(title="Répartition des données dans la variable : SeriousDlqin2yrs",
       x="SeriousDlqin2yrs", y="Frequency") + ylim(c(0,3500))

#============================ Creation d'une tache =============================
tache<-mlr::makeClassifTask(data=df , target="SeriousDlqin2yrs", positive="yes")
print(tache)

#==================== Échantillon apprentissage et validation =====================
set.seed(1234)
rs.holdout<-mlr::makeResampleInstance("Holdout", split=0.7, task=tache)
train_mod = df[rs.holdout$train.inds[[1]],] 
valid_mod = df[rs.holdout$test.inds[[1]],] 

saveRDS(train_mod , file="train_mod.rds")
saveRDS(valid_mod, file="valid_mod.rds")

#===================== À RECUPERER Hist train ===================================
attach(train)
cible=SeriousDlqin2yrs
ggplot(data=train, aes(x=train$SeriousDlqin2yrs, fill=SeriousDlqin2yrs)) +
  geom_histogram(binwidth=.5, position="dodge") + 
  labs(title="Echantillon d'apprentissage",x="SeriousDlqin2yrs", y="Frequency") 

#===================== À RECUPERER Hist valid ============================
attach(valid_mod)
cible=SeriousDlqin2yrs
ggplot(data=valid, aes(x=valid$SeriousDlqin2yrs, fill=cible)) +
  geom_histogram(binwidth=.5, position="dodge") + 
  labs(title="Echantillon de validation",x="SeriousDlqin2yrs", y="Frequency") 

#======================= modélisation ====================
train<-subset(train_mod, select = -c(SeriousDlqin2yrs))
valid<-subset(valid_mod, select = -c(SeriousDlqin2yrs))

saveRDS(train , file="train.rds")
saveRDS(valid, file="valid.rds")




#================================ LOGISTIC REGRESSION =============================
glm.fit <- glm(cible ~., data = train, family = binomial)
summary(glm.fit)
glm.probs <- predict(glm.fit, newdata = valid,type = "response")
glm.pred <- ifelse(glm.probs > 0.5, "yes", "no")

glm.tab = table(Actual = valid$cible, Predicted = glm.pred)
glm.tab

train_con_mat = confusionMatrix(glm.tab, positive = "yes")
c(train_con_mat$overall["Accuracy"], 
  train_con_mat$byClass["Sensitivity"], 
  train_con_mat$byClass["Specificity"])

plot(roc(valid$cible ~ glm.probs, plot = TRUE, print.auc = TRUE), print.auc = T,
     print.auc.y = 0.2, main='ROC curve - Logistic regression')

#============================== Random Forest =============================
set.seed(123)
bg.fit=randomForest(cible~., train)
bg.pred=predict(bg.fit, valid)
bg.prob=predict(bg.fit, valid, type='prob')
bg.tab = table(actual = valid$cible,predict = bg.pred)
bg.tab

train_con_mat1 = confusionMatrix(bg.tab, positive = "yes")
c(train_con_mat1$overall["Accuracy"], 
  train_con_mat1$byClass["Sensitivity"], 
  train_con_mat1$byClass["Specificity"])

plot(roc(valid$cible ~ bg.prob[,1], plot = TRUE, print.auc = TRUE), print.auc = T,
     print.auc.y = 0.2, main='ROC curve - Random Forest')

#============================== penalized regression: Lasso =================
set.seed(1234)
Xx <- data.matrix(subset( train , select =-c(cible)))
yy <- data.matrix(subset( train , select =c(cible)))
cv.lasso <- cv.glmnet(Xx, yy , alpha = 1 , family = "binomial")
plot(cv.lasso)

l.fit <- glmnet(Xx, yy, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min,standardize = TRUE)
coef(l.fit) 

new <- data.matrix(subset(valid, select=-c(cible)))
l.pred <- predict(l.fit, new, s=cv.lasso$lambda.min, type=c("class"))
l.prob <- predict(l.fit, new, s=cv.lasso$lambda.min, type=c("link"))

l.pred0 <- ifelse(l.pred=='1', "no", "yes")
l.tab= table(Actual = valid$cible, Predicted = l.pred0)
l.tab

train_con_mat0 = confusionMatrix(l.tab, positive = "yes")
c(train_con_mat0$overall["Accuracy"], 
  train_con_mat0$byClass["Sensitivity"], 
  train_con_mat0$byClass["Specificity"])

plot(roc(valid$cible ~ as.vector(l.prob), plot = TRUE, print.auc = TRUE), print.auc = T,
     print.auc.y = 0.2, main='ROC curve - penalized regression: Lasso ')

#============================== penalized regression: Ridge =================
set.seed(1234)
Xx <- data.matrix(subset( train , select =-c(cible)))
yy <- data.matrix(subset( train , select =c(cible)))
cv.ridge <- cv.glmnet(Xx, yy , alpha = 0 , family = "binomial")
plot(cv.ridge)

l.fit <- glmnet(Xx, yy, alpha = 0, family = "binomial", lambda = cv.ridge$lambda.min,standardize = TRUE)
coef(l.fit) 

new <- data.matrix(subset(valid, select=-c(cible)))
l.pred <- predict(l.fit, new, s=cv.ridge$lambda.min, type=c("class"))
l.prob <- predict(l.fit, new, s=cv.ridge$lambda.min, type=c("link"))

l.pred0 <- ifelse(l.pred=='1', "no", "yes")
l.tab= table(Actual = valid$cible, Predicted = l.pred0)
l.tab

train_con_mat0 = confusionMatrix(l.tab, positive = "yes")
c(train_con_mat0$overall["Accuracy"], 
  train_con_mat0$byClass["Sensitivity"], 
  train_con_mat0$byClass["Specificity"])

plot(roc(valid$cible ~ as.vector(l.prob), plot = TRUE, print.auc = TRUE), print.auc = T,
     print.auc.y = 0.2, main='ROC curve - penalized regression: Lasso ')

#============================== penalized regression: Elastic-Net =================
set.seed(1234)
Xx <- data.matrix(subset( train , select =-c(cible)))
yy <- data.matrix(subset( train , select =c(cible)))
cv.Elastic <- cv.glmnet(Xx, yy , alpha = 0.8 , family = "binomial")
plot(cv.Elastic)

l.fit <- glmnet(Xx, yy, alpha = 0.8, family = "binomial", lambda = cv.Elastic$lambda.min,standardize = TRUE)
coef(l.fit) 

new <- data.matrix(subset(valid, select=-c(cible)))
l.pred <- predict(l.fit, new, s=cv.Elastic$lambda.min, type=c("class"))
l.prob <- predict(l.fit, new, s=cv.Elastic$lambda.min, type=c("link"))

l.pred0 <- ifelse(l.pred=='1', "no", "yes")
l.tab= table(Actual = valid$cible, Predicted = l.pred0)
l.tab

train_con_mat0 = confusionMatrix(l.tab, positive = "yes")
c(train_con_mat0$overall["Accuracy"], 
  train_con_mat0$byClass["Sensitivity"], 
  train_con_mat0$byClass["Specificity"])

plot(roc(valid$cible ~ as.vector(l.prob), plot = TRUE, print.auc = TRUE), print.auc = T,
     print.auc.y = 0.2, main='ROC curve - penalized regression: Lasso ')



#============================ ROC synthese ============================

par(pty="s") 


lgROC <- roc(valid$cible ~ glm.probs,plot=TRUE,legacy.axes=TRUE, lwd=2, 
             col="blue",grid=TRUE, add = TRUE)

lassoROC <-roc(valid$cible ~ as.vector(l.prob),plot=TRUE,legacy.axes=TRUE, lwd=2, 
               col="yellow",grid=TRUE, add = TRUE)

rfROC <- roc(valid$cible ~ bg.prob[,1],plot=TRUE,legacy.axes=TRUE, lwd=2, 
             col="green",grid=TRUE, add = TRUE)
  
ridgeROC <- roc(valid$cible ~ svm.pred0,plot=TRUE,legacy.axes=TRUE, lwd=2, 
             col="pink",grid=TRUE, add = TRUE)

elasticROC <- roc(valid$SeriousDlqin2yrs, attributes(knn.fit)$prob,plot=TRUE,legacy.axes=TRUE, lwd=2, 
    col="orange",grid=TRUE, add = TRUE)


legend("bottomright",legend=c("PLTR","Logistic Regression","LR-lasso" ,"SVM","KNN"),
       col=c("red","blue", "yellow", "green", "pink", "orange"),
       lwd=4)

#==================================== FIN =================================
#==============================================================================

#library(pROC)
#par(pty="s") 
#lrROC <- roc(valid_knn$SeriousDlqin2yrs, attributes(knn.fit)$prob,plot=TRUE,print.auc=TRUE,col="green",lwd =4,legacy.axes=TRUE,main="ROC Curves")
## Setting levels: control = 0, case = 1
## Setting direction: controls < cases
#svmROC <- roc(vdata_Y ~ svm_predict,plot=TRUE,print.auc=TRUE,col="blue",lwd = 4,print.auc.y=0.4,legacy.axes=TRUE,add = TRUE)
## Setting levels: control = 0, case = 1 ## Setting direction: controls < cases
#legend("bottomright",legend=c("Logistic Regression","SVM"),col=c("green","blue"),lwd=4)

#plot(roc(valid$cible ~ psvm,plot=TRUE,print.auc=TRUE,col="blue",lwd = 4,print.auc.y=0.4,legacy.axes=TRUE,add = TRUE))


#library(ROSE)
#data_balanced <- ovun.sample(cible ~ ., data = newtrain, method = "under", p=0.2)$data
#table(data_balanced$cible)