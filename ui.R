#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinydashboard)
library(shinythemes)
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


# Define UI for application that draws a histogram
shinyUI(fluidPage(
    # Application title
    theme = shinytheme("united"),
    
    titlePanel("Big Data Analytics"),
    p("ADAMOU MAIKANOU Aminatou, EL HAJJAJI Jamal, SELMANE Houda"),
    
    ui<- navbarPage( title="",
                     tabPanel("Avant propos",  p("Grâce aux avancées technologiques, les établissements financiers réduisent les risques encourus à l’aide d’informations sur les clients. 
                             Des méthodes statistiques et des techniques d’apprentissage automatique (Machine Learning) 
                             permettent d’analyser ces données disponibles et de les résumer via un score de risque.
                             Ainsi, l’objectif de notre projet est d’évaluer ce score à l’aide des différentes méthodes de classification supervisée,
                             à savoir les méthodes de pénalisation et d’agrégation. Nous allons, par la suite, 
                              benchmarker ses modèles et les comparer entre elles à l’aide de leurs performances distinctes, pour en tirer le meilleur."),
                              p(strong("Méthode de pénalisation : ")), p("En présence de Big Data, les solutions analytiques traditionnelles font face à un problème majeur : les corrélations spécieuses.
Dans le modèle de regression linéaire, l’estimation du vecteur des paramètres par la méthode des MCO requiert que la matrice des variables explicatives soit de plein rang. Or, cette condition est violée en présence des fortes corrélations spécieuses entre variables inhérentes aux données massives. Ceci a pour conséquence une instabilité des paramètres estimés et donc un impact sur la stabilité des prévisions.
L’estimateur MCO, sous certaines hypothèses, est sans biais et de variance minimale dans la classe des estimateurs linéaires en y. La variance peut néanmoins être très importante, surtout lorsque le nombre de prédicteurs est élevé dans un contexte de Big data.
Ainsi, les méthodes de régression pénalisée se présente comme une solution face à ce problème : s’autoriser un peu de biais pour baisser de manière significative la variance et donc l’erreur de prévision. Elles permettent de réduire la variabilité de l’estimateur MCO via une régularisation des coefficients.
Dans ce projet, nous allons modéliser la probabilité de défaut d’abord par une régression Ridge, qui est historiquement la première méthode de pénalisation proposée, et par la suite, des régressions Lasso, Adaptative Lasso et Elastic-Net."), 
                              p(strong("Méthode d’agrégation :")), 
                              p("Les arbres de décisions représentent une suite de règles qui permet de résoudre un problème de classification ou de régression. Ils sont assez performants pour modéliser de manière non paramétrique des relations non linéaires et permettent de s’adapter à des données complexes.
Ils ont cependant des pouvoirs prédictifs limités et sont souvent sujet au sur- apprentissage. L’algorithme apprend avec beaucoup de précision les données d’entrainement qu’il ne parvient pas à généraliser un résultat satisfaisant sur de nouvelles données.
Les méthodes d’agrégation permettent de pallier à cette insuffisance. Il s’agit d’une technique consistant à entraîner plusieurs arbres pour ensuite considérer l'ensemble de leurs prédictions.
Dans ce projet, nous allons modéliser la probabilité de défaut en utilisant les deux classes de méthodes d’agrégation : le raffinement du Bagging via l’algorithme du Foret aléatoire et les méthodes de Boosting ( Adaboost et Gradient Boosting)")
                     ),
                     
                     tabPanel("Les donnees", 
                              p("On utilise le jeu de donnees  'Give me some credit'. 
              Ci-dessous un extrait du jeu de donnees:  "),
                              dataTableOutput('dataTable'),
                              h3("Description de la variable cible et des variables explicatives"),
                              p("SeriousDlqin2yrs(variable cible) : Binaire - si le consommateur fait de defaut de 
             credit(1), sinon (0)"),
                              p("1) RevolvingUtilizationOfUnsecuredLines  : Quantitative - Solde total des cartes de credit et marges de credit personnelles, sauf les biens immobiliers et les dettes a temperament comme les prets automobiles, divise par la somme des limites de credit"),
                              p("2) Age : Quantitative -Age de l'emprunteur (en annees)"),
                              p("3) NumberOfTime30-59DaysPastDueNotWorse: Quantitative -Nombre de fois ou un emprunteur a entre 30 et 59 jours en retard mais pas plus dans les deux dernieres annees"),
                              p("4) DebtRatio: Quantitative - Les paiements mensuels de la dette, la pension alimentaire et les frais de subsistance sur le revenu mensuel brut"),
                              p("5) MonthlyIncome : Quantitative - Revenus mensuels"),
                              p("6) NumberOfOpenCreditLinesAndLoans : Quantitative - Nombre de prets ouverts (comme un voiture ou un pret hypothecaire) et lignes de credit (cartes de credit)"),
                              p("7) NumberOfTimes90DaysLate : Quantitative - Nombre de fois ou un emprunteur a 90 jours ou plus de retard de paiement"),
                              p("8) NumberRealEstateLoansOrLines : Quantitative - Nombre de prets hypothecaires et immobiliers immobiliers, y compris les lignes de lignes de credit d'accession a la propriete"),
                              p("9) NumberOfTimes60-89DaysPastDueNotWorse: Quantitative - Nombre de fois ou un emprunteur a entre 60 et 89 jours en retard mais pas plus dans les deux dernieres annees"),
                              p("10) NumberOfDependents : Quantitative - Nombre de personnes a charge dans la famille a l exclusion d eux-memes (conjoint,enfants, etc.)"),
                              
                              fluidRow(
                                  column( width = 6 ,   h4("Statistique descriptive"),
                                          selectInput("var", p("Selectionner une variable:"), choices = colnames(brute), 
                                                      selected = " RevolvingUtilizationOfUnsecuredLines" ), verbatimTextOutput("var")
                                  ))),
                              

                     
                     tabPanel("Traitement de donnes", 
                              h4("Les donnees apres le nettoyage"),
                              p("Vous l'aurez remarque dans la section 'Les donnees' la presence de valeurs maquantes. Ainsi, nous avons 
             proceder au nettoyage de la base de donnees et nous sommes passer de 150000 observations A 120269 observations."),
                              p("Ci-dessous, un extrait de notre nouvelle base de donnees:"),
                              h5("Apres le nettoyage:"),
                              dataTableOutput('ClenTable'), 
                              
                              h4("Repartition de la variable cible 'SeriousDlqin2yrs'"), 
                              p("Une premiere analyse des frequences de notre variable de reponse, 
              ayant pour modalite 1 si l'individu a connu l'evenement et 0 sinon nous adresse le constat suivant : 
            91% des individus de la base de donnees n'ont pas connu l'evenement, 
             et seulement 9% sont des mauvais payeurs :"),
                              fluidRow(
                                  column( width = 6 , plotOutput(outputId = "hist1"))),
                              
                              p("Nous ne faisons pas face a une base de donnees avec des 
            donnees desequilibre (un probleme d'evenement rare). Danc, on passe directement au 
            partionnement du jeu de donnees."),
                              h4("Echantillon d'apprentissage & de validation"),
                              p("La premiere Etape consiste a partitionner notre echantillon en deux sous-echantillons : 
             un premier d'apprentissage, et un second de validation. 30% des observations initiales serviront à 
             construire l'echantillon de validation et les 70% restantes constitueront notre echantillon d'apprentissage. 
             C'est sur ce dernier que nous allons estimer notre modèle. Et, l'échanlillon de validation, nous permettra de comparer
             la perfomance des differentes methodes que nous verrons dans la section 'Modelisation'. 
             Nous pouvons donc voir que la proportion d'evenement ou de non-evenement, 
             dans ces deux echantillons est similaire :"),
                              fluidRow(
                                  column(width = 6 , plotOutput(outputId = "hist2")) ,
                                  column(width = 6, plotOutput(outputId = "hist3")))
                              
                     ),
                
                     
                     tabPanel("Modélisation", 
             
                              p("Dans cette section, nous allons créer des modèles pour chaque approche"),
                              tabsetPanel(
                                  tabPanel("Régression logistique", br(),
                                           p("La régression logistique est une méthode de Machine Learning 
                        typiquement utilisée lorsque la variable de réponse est qualitative. 
                        Son but est donc de relier la variable dépendante à des prédicteurs 
                        (quantitatifs et/ou qualitatifs)."),
                                           fluidRow(
                                               column( width = 6 , h5("Courbe de ROC"), 
                                                       plotOutput(outputId = "ROC4")),
                                               column(width = 6 , h5("Matrice de confusion"),     
                                                      br(), verbatimTextOutput(outputId = "mc4"))
                                           )),
                                  tabPanel("Lasso", br(),
                                           p("Il s'agit d'une pénalisation de la norme des coefficients MCO.
                        Elle ajoute un terme de pénalité à la fonction de coût.
                        Ce terme est la somme absolue des coefficients. 
                        Lorsque la valeur des coefficients augmente à partir de 0, 
                        ce terme pénalise, oblige le modèle, à diminuer la valeur des coefficients 
                        afin de réduire les pertes. 
                      "),
                                           fluidRow(
                                               column( width = 6 , h5("Courbe de ROC"),
                                                       plotOutput(outputId = "ROC5")),
                                               column(width = 6 , h5("Matrice de confusion"),     
                                                      br(), verbatimTextOutput(outputId = "mc5"))
                                           )),
                                  tabPanel("Random Forest", br(),
                                           p("Une forêt aléatoire est formée par de multiples arbres de décisions 
              estimés sur des échantillons bootstrapés. Plusieurs modèles faibles 
              sont alors aggrégés afin de créer un modèle robuste. 
              C'est un algorithme de Machine Learning particulièrement 
              efficace pour la prédiction."),
                                           fluidRow(
                                               column( width = 6 , h5("Courbe de ROC"),
                                                       plotOutput(outputId = "ROC1")),
                                               column(width = 6 , h5("Matrice de confusion"),     
                                                      br(), verbatimTextOutput(outputId = "mc1"))
                                           )),
                                  
                                  tabPanel("Ridge", br(),
                                           p("La différence entre la régression ridge et lasso est 
                      qu’elle a tendance à rendre les coefficients au zéro absolu 
                      par rapport à Ridge qui ne définit jamais la valeur du coefficient au zéro absolu."),
                                           fluidRow(
                                               column( width = 6 , h5("Courbe de ROC"), 
                                                       plotOutput(outputId = "ROC2")),
                                               column(width = 6 , h5("Matrice de confusion"),     
                                                      br(), verbatimTextOutput(outputId = "mc2") )
                                           )),
                                  tabPanel("Élastic-Net", br(),
                                           p("La régression au lasso peut provoquer un petit biais dans
                      le modèle où la prédiction dépend trop d’une variable particulière.
                      Dans ces cas, il est prouvé que le filet élastique combine mieux 
                      la régularisation du lasso et de la crête. L’avantage de cela n’élimine 
                      pas facilement le coefficient de colinéarité élevé."),
                                           fluidRow(
                                               column( width = 6 , h5("Courbe de ROC"), 
                                                       plotOutput(outputId = "ROC3")),
                                               column(width = 6 , h5("Matrice de confusion"),     
                                                      br(), plotOutput(outputId = "mc3")  )
                                           ))
                                  
                                 
                                           
                                       
                                           
                                         
                                           

                                           
                     
                                           
                                           
                                  
                              )
                              
                     ),
                     tabPanel("Conclusion", br(),
                              plotOutput(outputId = "ROCSynthese"),
                              p("Le Random Forest est le modèle qui possède les meilleurs capacités prédictives, avec une AUC de 80%."),
                              
                              p("La régression logistique nous a permis de faire un modèle convergent, 
             efficace et qui possède des bonnes capacités prédictives.
             Il est aisément interprétable "),
                              
                              p("Les Forets aléatoires sont des algorithmes de Machine Learning présentant
             des qualités certaines et se révèlent nettement plus efficace que la régression logistique 
             en termes de précision. Néanmoins, ils sont plus coûteux et 
             plus complexes à mettre en place. En plus, ils sont des boites noires :Il per en explicabilité ce qu’il gagne en précision. "),
                              
                              p("Pour se ramener à des résultats exploitables, des méthodes de Machine Learning 
             interprétables ont été utilisées pour expliquer la contribution de chaque 
             variable sur les prédictions du modèle .")
                              
                              
                              
                      
                              
                            )
                     
    ),
    
    
    # Show a plot of the generated distribution
    mainPanel(
        plotOutput("distPlot")
    )
))
