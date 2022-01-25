#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
library(ggplot2)
library(dplyr)
library(shiny)

library(gbm)
library(dplyr)
library(caret)

library(randomForest)
library(plotly)

library(ROCR)
library(pROC)

library(glmnet)


# Define server logic required to draw a histogram
shinyServer(function(input, output) {
    
    
    
    output$dataTable = renderDataTable({
      brute =brute
        
    })
    
    
    output$var <- renderPrint({
        summary(data[[input$var]])
    })
    
    output$ClenTable = renderDataTable({
        data=data
    })
    
    
    output$hist1 <- renderPlot({
        attach(data)
        cible=SeriousDlqin2yrs
        ggplot(data=data, aes(x=data$SeriousDlqin2yrs, fill=cible)) +
            geom_histogram(binwidth=.5, position="dodge") + 
            labs(title="Repartition des donnees dans la variable : SeriousDlqin2yrs",
                 x="SeriousDlqin2yrs", y="Frequency") + ylim(c(0,3500))
    })
    
    
    
    output$hist2 <- renderPlot( {
        attach(train)
        cible=SeriousDlqin2yrs
        ggplot(data=train, aes(x=train$SeriousDlqin2yrs, fill=cible)) +
            geom_histogram(binwidth=.5, position="dodge") + 
            labs(title="Echantillon d'apprentissage",x="SeriousDlqin2yrs", y="Frequency") 
    })
    
    
    output$hist2 <- renderPlot( {
        attach(valid)
        cible=SeriousDlqin2yrs
        ggplot(data=valid, aes(x=valid$SeriousDlqin2yrs, fill=cible)) +
            geom_histogram(binwidth=.5, position="dodge") + 
            labs(title="Echantillon de validation",x="SeriousDlqin2yrs", y="Frequency") 
    })
    
    ##Modelisation

    
    output$lambda <- renderPlot({
        cible=SeriousDlqin2yrs
        newtrain=train
        set.seed(1234)
        X <- data.matrix(subset( newtrain , select =-c(cible)))
        y <- data.matrix(subset( newtrain , select =c(cible)))
        cv.lasso <- cv.glmnet(X, y , alpha = 1 , family = "binomial")
        plot(cv.lasso)
    })
    
    r2 <- eventReactive(input$button1,{ 
        cible=SeriousDlqin2yrs
        newtrain=train
        set.seed(1234)
        X <- data.matrix(subset( newtrain , select =-c(cible)))
        y <- data.matrix(subset( newtrain , select =c(cible)))
        cv.lasso <- cv.glmnet(X, y , alpha = 1 , family = "binomial")
        cv.lasso$lambda.min
    })
    
    output$best <- renderPrint({
        r2()
    }) 
    

 
   
    
    
    output$mc <- renderPrint({
        cible=SeriousDlqin2yrs
        newtrain=train
        set.seed(1234)
        X <- data.matrix(subset( newtrain , select =-c(cible)))
        y <- data.matrix(subset( newtrain , select =c(cible)))
        cv.lasso <- cv.glmnet(X, y , alpha = 1 , family = "binomial")
        pltr.fit <- glmnet(X, y, alpha = 1, family = "binomial",lambda = cv.lasso$lambda.min,standardize = TRUE)
        newv <- data.matrix(subset( newvalid, select=-c(cible)))
        pltr.pred <- predict(pltr.fit, newv, s=cv.lasso$lambda.min, type=c("class"))
        
        pltr.pred0 <- ifelse(pltr.pred =='1', "no", "yes")
        pltr.tab = table(actual = newvalid$cible, predicted = pltr.pred0)
        pltr.tab
        
    })
    
    r3 <- eventReactive(input$button3,{ 
      attach(valid)
      attach(train)
        cible=SeriousDlqin2yrs
        newtrain=train
        set.seed(1234)
        X <- data.matrix(subset( newtrain , select =-c(cible)))
        y <- data.matrix(subset( newtrain , select =c(cible)))
        cv.lasso <- cv.glmnet(X, y , alpha = 1 , family = "binomial")
        pltr.fit <- glmnet(X, y, alpha = 1, family = "binomial",lambda = cv.lasso$lambda.min,standardize = TRUE)
        newv <- data.matrix(subset( newvalid, select=-c(cible)))
        pltr.pred <- predict(pltr.fit, newv, s=cv.lasso$lambda.min, type=c("class"))
        
        pltr.pred0 <- ifelse(pltr.pred =='1', "no", "yes")
        pltr.tab = table(actual = newvalid$cible, predicted = pltr.pred0)
        train_con_mat0 = confusionMatrix(pltr.tab, positive = "yes")
        c(train_con_mat0$overall["Accuracy"], 
          train_con_mat0$byClass["Sensitivity"], 
          train_con_mat0$byClass["Specificity"])
    })
    
 
    
    
    output$ROC <- renderPlot({
      attach(valid)
      attach(train)
        cible=SeriousDlqin2yrs
        newtrain=train
        set.seed(1234)
        X <- data.matrix(subset( newtrain , select =-c(cible)))
        y <- data.matrix(subset( newtrain , select =c(cible)))
        cv.lasso <- cv.glmnet(X, y , alpha = 1 , family = "binomial")
        pltr.fit <- glmnet(X, y, alpha = 1, family = "binomial",lambda = cv.lasso$lambda.min,standardize = TRUE)
        newv <- data.matrix(subset( newvalid, select=-c(cible)))
        pltr.prob <- predict(pltr.fit, newv, s=cv.lasso$lambda.min, type=c("link"))
        roc(newvalid$cible ~ as.vector(pltr.prob),plot=TRUE,legacy.axes=TRUE, lwd=2, 
            col="red",print.auc=TRUE,grid=TRUE, main='ROC curve - PLTR')
    })
    
    output$mc5 <- renderPrint({
        
        attach(valid)
        attach(train)
        set.seed(1234)
        Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
        yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
        cv.lasso <- cv.glmnet(Xx, yy , alpha = 1 , family = "binomial")
        l.fit <- glmnet(Xx, yy, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min,standardize = TRUE)
        new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
        l.pred <- predict(l.fit, new, s=cv.lasso$lambda.min, type=c("class"))
        l.prob <- predict(l.fit, new, s=cv.lasso$lambda.min, type=c("link"))
        l.pred0 <- ifelse(l.pred=='1', "no", "yes")
        l.tab= table(Actual = valid$SeriousDlqin2yrs, Predicted = l.pred0)
        l.tab
    })
    
    output$ROC5 <- renderPlot({
        cible=SeriousDlqin2yrs
        attach(valid)
        attach(train)
        set.seed(1234)
        Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
        yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
        cv.lasso <- cv.glmnet(Xx, yy , alpha = 1 , family = "binomial")
        l.fit <- glmnet(Xx, yy, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min,standardize = TRUE)
        new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
        l.prob <- predict(l.fit, new, s=cv.lasso$lambda.min, type=c("link"))
        roc(valid$SeriousDlqin2yrs ~ as.vector(l.prob),plot=TRUE,legacy.axes=TRUE, lwd=2, 
            col="red",print.auc=TRUE,grid=TRUE, main='ROC curve - Penalized regression: Lasso')
        
    })
    
    
    output$mc4 <- renderPrint({
      attach(valid)
      attach(train)
      glm.fit <- glm(SeriousDlqin2yrs~., data = train, family = binomial)
      glm.probs <- predict(glm.fit, newdata = valid,type = "response")
      glm.pred <- ifelse(glm.probs > 0.5, "yes", "no")
      glm.tab = table(Actual = valid$SeriousDlqin2yrs, Predicted = glm.pred)
      glm.tab
    })
    
    output$ROC4 <- renderPlot({ 
     attach(valid)
      attach(train)

        glm.fit <- glm(SeriousDlqin2yrs ~., data = train, family = binomial)
        glm.probs <- predict(glm.fit, newdata = valid,type = "response")
        roc(valid$SeriousDlqin2yrs ~ glm.probs,plot=TRUE,legacy.axes=TRUE, lwd=2, 
            col="red",print.auc=TRUE,grid=TRUE, main='ROC curve - Logistic regression')
    })
    
    output$mc1 <- renderPrint({
      set.seed(123)
      attach(valid)
      attach(train)
      bg.fit=randomForest(as.factor(SeriousDlqin2yrs)~., train)
      bg.pred=predict(bg.fit, valid)
      bg.prob=predict(bg.fit, valid, type='prob')
      bg.tab = table(Actual = valid$SeriousDlqin2yrs,Predicted = bg.pred)
      bg.tab
      
    })
    output$ROC1 <- renderPlot({
      attach(valid)
      attach(train)
        set.seed(123)
      set.seed(123)
      bg.fit=randomForest(as.factor(SeriousDlqin2yrs)~., train)
      bg.pred=predict(bg.fit, valid)
      bg.prob=predict(bg.fit, valid, type='prob')
      roc(valid$SeriousDlqin2yrs ~ bg.prob[,1],plot=TRUE,legacy.axes=TRUE, lwd=2, 
          col="red",print.auc=TRUE,grid=TRUE, main='ROC curve - Random Forest')
    })
    
    output$mc2 <- renderPrint({
      attach(valid)
      attach(train)
      set.seed(1234)
      Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
      yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
      cv.ridge <- cv.glmnet(Xx, yy , alpha = 0 , family = "binomial")
      l.fit <- glmnet(Xx, yy, alpha = 0, family = "binomial", lambda = cv.ridge$lambda.min,standardize = TRUE)
      new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
      l.pred <- predict(l.fit, new, s=cv.ridge$lambda.min, type=c("class"))
      l.prob <- predict(l.fit, new, s=cv.ridge$lambda.min, type=c("link"))
      l.pred0 <- ifelse(l.pred=='1', "no", "yes")
      l.tab= table(Actual = valid$SeriousDlqin2yrs, Predicted = l.pred0)
      l.tab
    })
    
    output$ROC2 <- renderPlot({ 
      attach(valid)
      attach(train)
      set.seed(1234)
      Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
      yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
      cv.ridge <- cv.glmnet(Xx, yy , alpha = 0 , family = "binomial")
      l.fit <- glmnet(Xx, yy, alpha = 0, family = "binomial", lambda = cv.ridge$lambda.min,standardize = TRUE)
      new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
      l.prob <- predict(l.fit, new, s=cv.ridge$lambda.min, type=c("link"))
      roc(valid$SeriousDlqin2yrs ~ as.vector(l.prob),plot=TRUE,legacy.axes=TRUE, lwd=2, 
          col="red",print.auc=TRUE,grid=TRUE, main='ROC curve - Penalized regression: Ridge')
    })
    
    output$mc3 <- renderPlot({
      attach(valid)
      attach(train)
      set.seed(1234)
      Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
      yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
      cv.Elastic <- cv.glmnet(Xx, yy , alpha = 0.8 , family = "binomial")
      l.fit <- glmnet(Xx, yy, alpha = 0.8, family = "binomial", lambda = cv.Elastic$lambda.min,standardize = TRUE)
      new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
      l.pred <- predict(l.fit, new, s=cv.Elastic$lambda.min, type=c("class"))
      l.prob <- predict(l.fit, new, s=cv.Elastic$lambda.min, type=c("link"))
      l.pred0 <- ifelse(l.pred=='1', "no", "yes")
      l.tab= table(Actual = valid$SeriousDlqin2yrs, Predicted = l.pred0)
      l.tab
    })
    output$ROC3 <- renderPlot({
      attach(valid)
      attach(train)
      set.seed(1234)
      Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
      yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
      cv.Elastic <- cv.glmnet(Xx, yy , alpha = 0.8 , family = "binomial")
      l.fit <- glmnet(Xx, yy, alpha = 0.8, family = "binomial", lambda = cv.Elastic$lambda.min,standardize = TRUE)
      new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
      l.prob <- predict(l.fit, new, s= cv.Elastic$lambda.min, type=c("link"))
      roc(valid$SeriousDlqin2yrs ~ as.vector(l.prob),plot=TRUE,legacy.axes=TRUE, lwd=2, 
          col="red",print.auc=TRUE,grid=TRUE, main='ROC curve - Penalized regression: Elastc-Net')
     
        
    })
    
    output$ROCSynthese <- renderPlot({
      newtrain=train
      newvalid=valid
       
        glm.fit <- glm(SeriousDlqin2yrs ~., data = train, family = binomial)
        glm.probs <- predict(glm.fit, newdata = valid,type = "response")
        
        set.seed(1234)
        Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
        yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
        cv.lasso <- cv.glmnet(Xx, yy , alpha = 1 , family = "binomial")
        l.fit <- glmnet(Xx, yy, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min,standardize = TRUE)
        new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
        l.prob <- predict(l.fit, new, s=cv.lasso$lambda.min, type=c("link"))
        
        set.seed(123)
        bg.fit=randomForest(as.factor(SeriousDlqin2yrs)~., train)
        bg.pred=predict(bg.fit, valid)
        bg.prob=predict(bg.fit, valid, type='prob')
        
        set.seed(1234)
        Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
        yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
        cv.ridge <- cv.glmnet(Xx, yy , alpha = 0 , family = "binomial")
        l.fit <- glmnet(Xx, yy, alpha = 0, family = "binomial", lambda =  cv.ridge$lambda.min,standardize = TRUE)
        new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
        l.prob <- predict(l.fit, new, s= cv.ridge$lambda.min, type=c("link"))
        
        
        set.seed(1234)
        Xx <- data.matrix(subset( train , select =-c(SeriousDlqin2yrs)))
        yy <- data.matrix(subset( train , select =c(SeriousDlqin2yrs)))
        cv.Elactic <- cv.glmnet(Xx, yy , alpha = 0.8 , family = "binomial")
        l.fit <- glmnet(Xx, yy, alpha = 0.8, family = "binomial", lambda =cv.Elactic$lambda.min,standardize = TRUE)
        new <- data.matrix(subset(valid, select=-c(SeriousDlqin2yrs)))
        l.prob <- predict(l.fit, new, s=cv.Elactic$lambda.min, type=c("link"))
        
        
        
        
    
        
        
        
        par(pty="s") 
  
        lgROC <- roc(valid$SeriousDlqin2yrs ~ glm.probs,plot=TRUE,legacy.axes=TRUE, lwd=2, 
                     col="blue",grid=TRUE, add = TRUE)
        lassoROC <-roc(valid$SeriousDlqin2yrs ~ as.vector(l.prob),plot=TRUE,legacy.axes=TRUE, lwd=2, 
                       col="yellow",grid=TRUE, add = TRUE)
        rfROC <- roc(valid$SeriousDlqin2yrs ~ bg.prob[,1],plot=TRUE,legacy.axes=TRUE, lwd=2, 
                     col="green",grid=TRUE, add = TRUE)
        RidgeROC <- roc(valid$SeriousDlqin2yrs ~ as.vector(l.prob),plot=TRUE,legacy.axes=TRUE, lwd=2, 
                        col="orange",grid=TRUE, add = TRUE)
        ElasticROC <- roc(valid$SeriousDlqin2yrs ~ as.vector(l.prob),plot=TRUE,legacy.axes=TRUE, lwd=2, 
                          col="pink",grid=TRUE, add = TRUE)
        legend("bottomright",legend=c("Logistic Regression","LR-lasso","Random Forest", "LR-Ridge", "LR-Elastic-Net"),
               col=c("blue", "yellow", "green", "orange", "pink"),
               lwd=4)
        
    })
})
