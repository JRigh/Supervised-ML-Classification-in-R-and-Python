#---------------------------------
# Machine Learning algorithms in R
# Classification algorithms
#---------------------------------

library(tidyverse)     # dpylr, ggplot2,...
library(xtable)        # to export tables in LaTeX
data(iris)             # to load the data

#-----------------
# 0. Visualization
#-------------------------------------------------------------------------------

library(GGally)

# 1. multiple plots 
ggpairs(iris, ggplot2::aes(colour = Species, alpha = 0.4)) +
  labs(title = 'Summary of distributions',
       subtitle = 'Complete Iris dataset',
       y="", x="") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=9, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

#-------------------------------------------------------------------------------

# 1. splitting the dataset into training and test sets
set.seed(2023)
ind <- sample(2, nrow(iris),replace=TRUE,prob=c(0.7,0.3))
training <- iris[ind==1,]
testing <- iris[ind==2,]

# 2. save entire dataset, training and testing datasets
# save a copy of the dataset in .csv
write.csv(iris, 
          "C:/Users/julia/OneDrive/Desktop/github/9. Machine_learning_toolbox_R/iris.csv",
          row.names = FALSE)
write.csv(training, 
          "C:/Users/julia/OneDrive/Desktop/github/9. Machine_learning_toolbox_R/iris_training.csv",
          row.names = FALSE)
write.csv(testing, 
          "C:/Users/julia/OneDrive/Desktop/github/9. Machine_learning_toolbox_R/iris_testing.csv",
          row.names = FALSE)

#-----------------------------------------------------------
# 1.1 Naive Bayes classifier (multiple class classification)
#-----------------------------------------------------------

library(klaR) 

# 1. Build a Naive Bayes Classifier
set.seed(2023)
nb_model <- NaiveBayes(Species ~ .,         # train Naïve Bayes model
                       usekernel = FALSE,   # if true, a KDE is used to estimate the probabilities
                       data=training)       # on training model

pred_nb <- predict(nb_model, testing) # apply Naïve Bayes model on test set
pred_nb_training <- predict(nb_model, training) # apply Naïve Bayes model on train set

# 2. Create accuracy metrics table and coonfusion Matrix
tab_training <- table(pred_nb_training$class, training$Species)
tab_testing <- table(pred_nb$class, testing$Species)
result_nb_training <- caret::confusionMatrix(tab_training)
result_nb_testing <- caret::confusionMatrix(tab_testing)
result_nb_testing

# metrics on the training set
sensitivity_nb_training <- round(result_nb_training$byClass[, 1]*100, 2)
specificity_nb_training <- round(result_nb_training$byClass[, 2]*100, 2)
accuracy_nb_training <- round(result_nb_training$overall[1]*100, 2)
precision_nb_training <- round(mean(result_nb_training$byClass[,'Pos Pred Value'] *100), 2)
recall_nb_training <- round(mean(result_nb_training$byClass[, 'Sensitivity']*100), 2)
f1_score_nb_training <- round(2 * ((precision_nb_training * recall_nb_training) / (precision_nb_training + recall_nb_training)),2)

# metrics on the testing set
sensitivity_nb_testing <- round(result_nb_testing$byClass[, 1]*100, 2)
specificity_nb_testing <- round(result_nb_testing$byClass[, 2]*100, 2)
accuracy_nb_testing <- round(result_nb_testing$overall[1]*100, 2)
precision_nb_testing <- round(mean(result_nb_testing$byClass[,'Pos Pred Value'] *100), 2)
recall_nb_testing <-  round(mean(result_nb_testing$byClass[, 'Sensitivity']*100), 2)
f1_score_nb_testing <- round(2 * ((precision_nb_testing * recall_nb_testing) / (precision_nb_testing + recall_nb_testing)),2)


table_metrics <- matrix(c(accuracy_nb_training, accuracy_nb_testing,
                          precision_nb_training, precision_nb_testing,
                          recall_nb_training, recall_nb_testing, 
                          f1_score_nb_training, f1_score_nb_testing), byrow = TRUE, ncol = 2)
colnames(table_metrics) <-  c('Training', 'Testing')
rownames(table_metrics) <- c('accuracy', 'precision', 'recall', 'f1-score')
table_metrics

# export the results in LaTex document
print(xtable(table_metrics, type = "latex", digits=2), file = "tables.tex")


# 3.1 plot the confusion matrix
testing$pred_nb <- pred_nb$class
ggplot(testing, aes(Species, pred_nb, color = Species)) +
  geom_jitter(width = 0.2, height = 0.1, size=2) +
  annotate('text', label = paste("Specificity (Setosa) =", specificity_nb[1], ", Sensitivity (Setosa) =", sensitivity_nb[1]
  ), x = 2, y = 0.75, size = 3) + 
  annotate('text', label = paste("Specificity (Versicolor) =", specificity_nb[2], ", Sensitivity (Versicolor) =", sensitivity_nb[2]
  ), x = 2, y = 0.65, size = 3) +
  annotate('text', label = paste("Specificity (Virginica) =", specificity_nb[3], ", Sensitivity (Virginica) =", sensitivity_nb[3]
  ), x = 2, y = 0.55, size = 3) +
  labs(title = 'Confusion Matrix - Naive Bayes Classifier',
       subtitle = paste('Predicted vs. Observed from Iris dataset. Accuracy: ', accuracy_nb),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

# alternative
# 3.2 3lot the Confusion Matrix
cm_nb <- caret::confusionMatrix(factor(pred_nb$class), factor(testing$Species), dnn = c("Predicted", "Observed"))
plt <- as.data.frame(cm_nb$table)
plt$Predicted <- factor(plt$Predicted, levels=rev(levels(plt$Predicted)))

ggplot(plt, aes(Predicted, Observed, fill= Freq)) +
  geom_tile() + 
  geom_text(aes(label=Freq), size = 5) +
  scale_fill_gradient(low="white", high="darkgreen") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c(levels(factor(testing$Species)))) +
  scale_y_discrete(labels=rev(c(levels(factor(testing$Species))))) +
  labs(title = 'Confusion Matrix - Naive Bayes Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_nb, '%'),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

#--------------------------------------------------
# 1.2 Random Forest (multiple class classification)
#--------------------------------------------------

library(randomForest)

# 1. Build a Random Forest learning tree Classifier
set.seed(2023)
rf_model <- randomForest(Species~., 
                         data=training, 
                         ntree=1000,
                         proximity=TRUE) # train RF model

pred_rf <- predict(rf_model, testing) # apply RF model on test set
pred_rf_training <- predict(rf_model, training) # apply RF model on train set

# 2. Create accuracy metrics table and coonfusion Matrix
tab_training <- table(pred_rf_training, training$Species)
tab_testing <- table(pred_rf, testing$Species)
result_rf_training <- caret::confusionMatrix(tab_training)
result_rf_testing <- caret::confusionMatrix(tab_testing)
result_rf_testing

# metrics on the training set
sensitivity_rf_training <- round(result_rf_training$byClass[, 1]*100, 2)
specificity_rf_training <- round(result_rf_training$byClass[, 2]*100, 2)
accuracy_rf_training <- round(result_rf_training$overall[1]*100, 2)
precision_rf_training <- round(mean(result_rf_training$byClass[,'Pos Pred Value'] *100), 2)
recall_rf_training <- round(mean(result_rf_training$byClass[, 'Sensitivity']*100), 2)
f1_score_rf_training <- round(2 * ((precision_rf_training * recall_rf_training) / (precision_rf_training + recall_rf_training)),2)

# metrics on the testing set
sensitivity_rf_testing <- round(result_rf_testing$byClass[, 1]*100, 2)
specificity_rf_testing <- round(result_rf_testing$byClass[, 2]*100, 2)
accuracy_rf_testing <- round(result_rf_testing$overall[1]*100, 2)
precision_rf_testing <- round(mean(result_rf_testing$byClass[,'Pos Pred Value'] *100), 2)
recall_rf_testing <-  round(mean(result_rf_testing$byClass[, 'Sensitivity']*100), 2)
f1_score_rf_testing <- round(2 * ((precision_rf_testing * recall_rf_testing) / (precision_rf_testing + recall_rf_testing)),2)


table_metrics <- matrix(c(accuracy_rf_training, accuracy_rf_testing,
                          precision_rf_training, precision_rf_testing,
                          recall_rf_training, recall_rf_testing, 
                          f1_score_rf_training, f1_score_rf_testing), byrow = TRUE, ncol = 2)
colnames(table_metrics) <-  c('Training', 'Testing')
rownames(table_metrics) <- c('accuracy', 'precision', 'recall', 'f1-score')
table_metrics

# export the results in LaTex document
print(xtable(table_metrics, type = "latex", digits=2), file = "tables.tex")

# 3.1 plot the confusion matrix
table(predict(rf_model),training$Species)
pred_rf <- predict(rf_model, testing)
tab <- table(pred_rf, testing$Species)
result_rf <- caret::confusionMatrix(tab)
result_rf
testing$pred_rf <- pred_rf
ggplot(testing, aes(Species, pred_rf, color = Species)) +
  geom_jitter(width = 0.2, height = 0.1, size=2) +
  labs(title = 'Confusion Matrix - Random forest Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_rf_testing),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

# alternative
# 3.2 3lot the Confusion Matrix
cm_rf <- caret::confusionMatrix(factor(pred_rf), factor(testing$Species), dnn = c("Predicted", "Observed"))
plt <- as.data.frame(cm_rf$table)
plt$Predicted <- factor(plt$Predicted, levels=rev(levels(plt$Predicted)))

ggplot(plt, aes(Predicted, Observed, fill= Freq)) +
  geom_tile() + 
  geom_text(aes(label=Freq), size = 5) +
  scale_fill_gradient(low="white", high="darkgreen") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c(levels(factor(testing$Species)))) +
  scale_y_discrete(labels=rev(c(levels(factor(testing$Species))))) +
  labs(title = 'Confusion Matrix - Random Forest Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_rf_testing, '%'),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

#--------------------------------------------------------------------
# 1.3 Multinomial Logistic regression (multiple class classification)
#--------------------------------------------------------------------

library(stats4) 
library(splines) 
library(VGAM) 

# 1. Build a Multinomial Logistic Regression Classifier
set.seed(2023)
mlr_model <- vglm(Species ~ ., family=multinomial, training)

pred_mlr_training<- predict(mlr_model, training, type = "response")
pred_mlr_testing<- predict(mlr_model, testing, type = "response")
predictions <- apply(pred_mlr_testing, 1, which.max)
predictions_training <- apply(pred_mlr_training, 1, which.max)

# 2. Create accuracy metrics table and coonfusion Matrix
predictions[which(predictions=="1")] <- levels(iris$Species)[1]
predictions[which(predictions=="2")] <- levels(iris$Species)[2]
predictions[which(predictions=="3")] <- levels(iris$Species)[3]

predictions_training[which(predictions_training=="1")] <- levels(iris$Species)[1]
predictions_training[which(predictions_training=="2")] <- levels(iris$Species)[2]
predictions_training[which(predictions_training=="3")] <- levels(iris$Species)[3]


tab_training <- table(predictions_training, training$Species)
tab_testing <- table(predictions, testing$Species)
result_mlr_training <- caret::confusionMatrix(tab_training)
result_mlr_testing <- caret::confusionMatrix(tab_testing)
result_mlr_testing

# metrics on the training set
sensitivity_mlr_training <- round(result_mlr_training$byClass[, 1]*100, 2)
specificity_mlr_training <- round(result_mlr_training$byClass[, 2]*100, 2)
accuracy_mlr_training <- round(result_mlr_training$overall[1]*100, 2)
precision_mlr_training <- round(mean(result_mlr_training$byClass[,'Pos Pred Value'] *100), 2)
recall_mlr_training <- round(mean(result_mlr_training$byClass[, 'Sensitivity']*100), 2)
f1_score_mlr_training <- round(2 * ((precision_mlr_training * recall_mlr_training) / (precision_mlr_training + recall_mlr_training)),2)

# metrics on the testing set
sensitivity_mlr_testing <- round(result_mlr_testing$byClass[, 1]*100, 2)
specificity_mlr_testing <- round(result_mlr_testing$byClass[, 2]*100, 2)
accuracy_mlr_testing <- round(result_mlr_testing$overall[1]*100, 2)
precision_mlr_testing <- round(mean(result_mlr_testing$byClass[,'Pos Pred Value'] *100), 2)
recall_mlr_testing <-  round(mean(result_mlr_testing$byClass[, 'Sensitivity']*100), 2)
f1_score_mlr_testing <- round(2 * ((precision_mlr_testing * recall_mlr_testing) / (precision_mlr_testing + recall_mlr_testing)),2)


table_metrics <- matrix(c(accuracy_mlr_training, accuracy_mlr_testing,
                          precision_mlr_training, precision_mlr_testing,
                          recall_mlr_training, recall_mlr_testing, 
                          f1_score_mlr_training, f1_score_mlr_testing), byrow = TRUE, ncol = 2)
colnames(table_metrics) <-  c('Training', 'Testing')
rownames(table_metrics) <- c('accuracy', 'precision', 'recall', 'f1-score')
table_metrics

# export the results in LaTex document
print(xtable(table_metrics, type = "latex", digits=2), file = "tables.tex")

# 3.1 plot the Confusion Matrix
testing$pred_mlr <- predictions
ggplot(testing, aes(Species, pred_mlr, color = Species)) +
  geom_jitter(width = 0.2, height = 0.1, size=2) +
  labs(title = 'Confusion Matrix - Multinomial Logistic Regression',
       subtitle = paste('Predicted vs. Observed from Iris dataset. Accuracy: '),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

# alternative
# 3.2 3lot the Confusion Matrix
testing$pred_mlr <- predictions
cm_lrm <- caret::confusionMatrix(factor(testing$pred_mlr), factor(testing$Species), dnn = c("Predicted", "Observed"))
plt <- as.data.frame(cm_lrm$table)
plt$Predicted <- factor(plt$Predicted, levels=rev(levels(plt$Predicted)))

ggplot(plt, aes(Predicted, Observed, fill= Freq)) +
  geom_tile() + 
  geom_text(aes(label=Freq), size = 5) +
  scale_fill_gradient(low="white", high="darkgreen") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c(levels(factor(testing$Species)))) +
  scale_y_discrete(labels=rev(c(levels(factor(testing$Species))))) +
  labs(title = 'Confusion Matrix - Multinomial Logistic Regression Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_mlr_testing, '%'),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))


#------------------------------------------------------------
# 1.4 Support vector machines (multiple class classification)
#------------------------------------------------------------

library(e1071)

# 1. Build a Support Vector Machines Classifier
set.seed(2023)
svm_model <- svm(Species ~ ., 
                 data=training,
                 kernel="radial")                 #linear/polynomial/sigmoid

pred_svm <- predict(svm_model, testing)           # apply svm model on test set
pred_svm_training <- predict(svm_model, training) # apply svm model on train set

# 2. Create accuracy metrics table and coonfusion Matrix
tab_training <- table(pred_svm_training, training$Species)
tab_testing <- table(pred_svm, testing$Species)
result_svm_training <- caret::confusionMatrix(tab_training)
result_svm_testing <- caret::confusionMatrix(tab_testing)
result_svm_testing

# metrics on the training set
sensitivity_svm_training <- round(result_svm_training$byClass[, 1]*100, 2)
specificity_svm_training <- round(result_svm_training$byClass[, 2]*100, 2)
accuracy_svm_training <- round(result_svm_training$overall[1]*100, 2)
precision_svm_training <- round(mean(result_svm_training$byClass[,'Pos Pred Value'] *100), 2)
recall_svm_training <- round(mean(result_svm_training$byClass[, 'Sensitivity']*100), 2)
f1_score_svm_training <- round(2 * ((precision_svm_training * recall_svm_training) / (precision_svm_training + recall_svm_training)),2)

# metrics on the testing set
sensitivity_svm_testing <- round(result_svm_testing$byClass[, 1]*100, 2)
specificity_svm_testing <- round(result_svm_testing$byClass[, 2]*100, 2)
accuracy_svm_testing <- round(result_svm_testing$overall[1]*100, 2)
precision_svm_testing <- round(mean(result_svm_testing$byClass[,'Pos Pred Value'] *100), 2)
recall_svm_testing <-  round(mean(result_svm_testing$byClass[, 'Sensitivity']*100), 2)
f1_score_svm_testing <- round(2 * ((precision_svm_testing * recall_svm_testing) / (precision_svm_testing + recall_svm_testing)),2)


table_metrics <- matrix(c(accuracy_svm_training, accuracy_svm_testing,
                          precision_svm_training, precision_svm_testing,
                          recall_svm_training, recall_svm_testing, 
                          f1_score_svm_training, f1_score_svm_testing), byrow = TRUE, ncol = 2)
colnames(table_metrics) <-  c('Training', 'Testing')
rownames(table_metrics) <- c('accuracy', 'precision', 'recall', 'f1-score')
table_metrics

# export the results in LaTex document
print(xtable(table_metrics, type = "latex", digits=2), file = "tables.tex")

# 3.1 plot the confusion matrix
table(predict(svm_model),training$Species)
pred_svm <- predict(svm_model, testing)
tab <- table(pred_svm, testing$Species)
result_svm <- caret::confusionMatrix(tab)
result_svm
testing$pred_svm <- pred_svm
ggplot(testing, aes(Species, pred_svm, color = Species)) +
  geom_jitter(width = 0.2, height = 0.1, size=2) +
  labs(title = 'Confusion Matrix - Support Vector Machines Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_svm_testing),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

# alternative
# 3.2 3lot the Confusion Matrix
cm_svm <- caret::confusionMatrix(factor(pred_svm), factor(testing$Species), dnn = c("Predicted", "Observed"))
plt <- as.data.frame(cm_svm$table)
plt$Predicted <- factor(plt$Predicted, levels=rev(levels(plt$Predicted)))

ggplot(plt, aes(Predicted, Observed, fill= Freq)) +
  geom_tile() + 
  geom_text(aes(label=Freq), size = 5) +
  scale_fill_gradient(low="white", high="darkgreen") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c(levels(factor(testing$Species)))) +
  scale_y_discrete(labels=rev(c(levels(factor(testing$Species))))) +
  labs(title = 'Confusion Matrix - Support Vector Machines Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_svm_testing, '%'),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))


#---------------------------------------------------
# 1.5 Neural Network (multiple class classification)
#---------------------------------------------------

library(neuralnet)

# 1. Build a Neural Network Classifier
set.seed(2023)
iris$setosa <- iris$Species=="setosa"
iris$virginica <- iris$Species == "virginica"
iris$versicolor <- iris$Species == "versicolor"

# spliting into test and training again because we added variables
set.seed(2023)
ind <- sample(2, nrow(iris),replace=TRUE,prob=c(0.7,0.3))
training <- iris[ind==1,]
testing <- iris[ind==2,]

nn_model <- neuralnet(setosa+versicolor+virginica ~ 
                        Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                      data=training, hidden=c(10,10), rep = 5, err.fct = "ce", 
                      linear.output = F, lifesign = "minimal", stepmax = 1000000,
                      threshold = 0.001)

pred_nn_training<- predict(nn_model, training, type = "class")
pred_nn_testing<- predict(nn_model, testing, type = "class")
predictions <- apply(pred_nn_testing, 1, which.max)
predictions_training <- apply(pred_nn_training, 1, which.max)

# 2. Create accuracy metrics table and coonfusion Matrix
predictions[which(predictions=="1")] <- levels(iris$Species)[1]
predictions[which(predictions=="2")] <- levels(iris$Species)[2]
predictions[which(predictions=="3")] <- levels(iris$Species)[3]

predictions_training[which(predictions_training=="1")] <- levels(iris$Species)[1]
predictions_training[which(predictions_training=="2")] <- levels(iris$Species)[2]
predictions_training[which(predictions_training=="3")] <- levels(iris$Species)[3]


tab_training <- table(predictions_training, training$Species)
tab_testing <- table(predictions, testing$Species)
result_nn_training <- caret::confusionMatrix(tab_training)
result_nn_testing <- caret::confusionMatrix(tab_testing)
result_nn_testing

# metrics on the training set
sensitivity_nn_training <- round(result_nn_training$byClass[, 1]*100, 2)
specificity_nn_training <- round(result_nn_training$byClass[, 2]*100, 2)
accuracy_nn_training <- round(result_nn_training$overall[1]*100, 2)
precision_nn_training <- round(mean(result_nn_training$byClass[,'Pos Pred Value'] *100), 2)
recall_nn_training <- round(mean(result_nn_training$byClass[, 'Sensitivity']*100), 2)
f1_score_nn_training <- round(2 * ((precision_nn_training * recall_nn_training) / (precision_nn_training + recall_nn_training)),2)

# metrics on the testing set
sensitivity_nn_testing <- round(result_nn_testing$byClass[, 1]*100, 2)
specificity_nn_testing <- round(result_nn_testing$byClass[, 2]*100, 2)
accuracy_nn_testing <- round(result_nn_testing$overall[1]*100, 2)
precision_nn_testing <- round(mean(result_nn_testing$byClass[,'Pos Pred Value'] *100), 2)
recall_nn_testing <-  round(mean(result_nn_testing$byClass[, 'Sensitivity']*100), 2)
f1_score_nn_testing <- round(2 * ((precision_nn_testing * recall_nn_testing) / (precision_nn_testing + recall_nn_testing)),2)


table_metrics <- matrix(c(accuracy_nn_training, accuracy_nn_testing,
                          precision_nn_training, precision_nn_testing,
                          recall_nn_training, recall_nn_testing, 
                          f1_score_nn_training, f1_score_nn_testing), byrow = TRUE, ncol = 2)
colnames(table_metrics) <-  c('Training', 'Testing')
rownames(table_metrics) <- c('accuracy', 'precision', 'recall', 'f1-score')
table_metrics

# export the results in LaTex document
print(xtable(table_metrics, type = "latex", digits=2), file = "tables.tex")

# 3.1 plot the Confusion Matrix
testing$pred_nn <- predictions
ggplot(testing, aes(Species, pred_nn, color = Species)) +
  geom_jitter(width = 0.2, height = 0.1, size=2) +
  labs(title = 'Confusion Matrix - Neural Networks classifier',
       subtitle = paste('Predicted vs. Observed from Iris dataset. Accuracy: '),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

# alternative
# 3.2 3lot the Confusion Matrix
testing$pred_nn <- predictions
cm_lrm <- caret::confusionMatrix(factor(testing$pred_nn), factor(testing$Species), dnn = c("Predicted", "Observed"))
plt <- as.data.frame(cm_lrm$table)
plt$Predicted <- factor(plt$Predicted, levels=rev(levels(plt$Predicted)))

ggplot(plt, aes(Predicted, Observed, fill= Freq)) +
  geom_tile() + 
  geom_text(aes(label=Freq), size = 5) +
  scale_fill_gradient(low="white", high="darkgreen") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c(levels(factor(testing$Species)))) +
  scale_y_discrete(labels=rev(c(levels(factor(testing$Species))))) +
  labs(title = 'Confusion Matrix - Neural Networks Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_nn_testing, '%'),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

#--------------------------------------------------
# 1.6 Decision tree (multiple class classification)
#--------------------------------------------------

library(rpart)

# 1. Build a Decision Tree Classifier
set.seed(2023)
dt_model <- rpart(Species ~.,
                  data = training,
                  method = "class",
                  control = rpart.control(cp = 0),
                  parms = list(split = "information"))

pred_dt_training<- predict(dt_model, training, type = "class")
pred_dt_testing<- predict(dt_model, testing, type = "class")

# 2. Create accuracy metrics table and coonfusion Matrix
tab_training <- table(pred_dt_training, training$Species)
tab_testing <- table(pred_dt_testing, testing$Species)
result_dt_training <- caret::confusionMatrix(tab_training)
result_dt_testing <- caret::confusionMatrix(tab_testing)
result_dt_testing

# metrics on the training set
sensitivity_dt_training <- round(result_dt_training$byClass[, 1]*100, 2)
specificity_dt_training <- round(result_dt_training$byClass[, 2]*100, 2)
accuracy_dt_training <- round(result_dt_training$overall[1]*100, 2)
precision_dt_training <- round(mean(result_dt_training$byClass[,'Pos Pred Value'] *100), 2)
recall_dt_training <- round(mean(result_dt_training$byClass[, 'Sensitivity']*100), 2)
f1_score_dt_training <- round(2 * ((precision_dt_training * recall_dt_training) / (precision_dt_training + recall_dt_training)),2)

# metrics on the testing set
sensitivity_dt_testing <- round(result_dt_testing$byClass[, 1]*100, 2)
specificity_dt_testing <- round(result_dt_testing$byClass[, 2]*100, 2)
accuracy_dt_testing <- round(result_dt_testing$overall[1]*100, 2)
precision_dt_testing <- round(mean(result_dt_testing$byClass[,'Pos Pred Value'] *100), 2)
recall_dt_testing <-  round(mean(result_dt_testing$byClass[, 'Sensitivity']*100), 2)
f1_score_dt_testing <- round(2 * ((precision_dt_testing * recall_dt_testing) / (precision_dt_testing + recall_dt_testing)),2)


table_metrics <- matrix(c(accuracy_dt_training, accuracy_dt_testing,
                          precision_dt_training, precision_dt_testing,
                          recall_dt_training, recall_dt_testing, 
                          f1_score_dt_training, f1_score_dt_testing), byrow = TRUE, ncol = 2)
colnames(table_metrics) <-  c('Training', 'Testing')
rownames(table_metrics) <- c('accuracy', 'precision', 'recall', 'f1-score')
table_metrics

# export the results in LaTex document
print(xtable(table_metrics, type = "latex", digits=2), file = "tables.tex")

# 3.1 plot the confusion matrix
testing$pred_dt <- pred_dt_testing
ggplot(testing, aes(Species, pred_dt_testing, color = Species)) +
  geom_jitter(width = 0.2, height = 0.1, size=2) +
  labs(title = 'Confusion Matrix - Decision Tree Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_dt_testing),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

# alternative
# 3.2 3lot the Confusion Matrix
cm_dt <- caret::confusionMatrix(factor(pred_dt_testing), factor(testing$Species), dnn = c("Predicted", "Observed"))
plt <- as.data.frame(cm_dt$table)
plt$Predicted <- factor(plt$Predicted, levels=rev(levels(plt$Predicted)))

ggplot(plt, aes(Predicted, Observed, fill= Freq)) +
  geom_tile() + 
  geom_text(aes(label=Freq), size = 5) +
  scale_fill_gradient(low="white", high="darkgreen") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c(levels(factor(testing$Species)))) +
  scale_y_discrete(labels=rev(c(levels(factor(testing$Species))))) +
  labs(title = 'Confusion Matrix - Decision Tree Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_dt_testing, '%'),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

#--------------------------------------------
# 1.7 XGBoost (multiple class classification)
#--------------------------------------------

library(xgboost)

# 0. splitting the dataset into training and test sets
set.seed(2023)
ind <- sample(2, nrow(iris),replace=TRUE,prob=c(0.7,0.3))
training <- iris[ind==1,]
testing <- iris[ind==2,]

xgb_training = xgb.DMatrix(data = as.matrix(training[,-5]), label = training[,5])
xgb_testing = xgb.DMatrix(data = as.matrix(testing[,-5]), label = testing[,5])

# 1. Build a XGBoost Classifier
set.seed(2023)
xgb_model <- xgboost(data=xgb_training, max.depth=3, nrounds=50)

pred_xgb_training <- predict(xgb_model, xgb_training)
pred_y_xgb_training = as.factor((levels(training[,5]))[round(pred_xgb_training)])

pred_xgb_testing <- predict(xgb_model, xgb_testing)
pred_y_xgb_testing = as.factor((levels(testing[,5]))[round(pred_xgb_testing)])

# 2. Create accuracy metrics table and coonfusion Matrix
tab_training <- table(pred_y_xgb_training, training$Species)
tab_testing <- table(pred_y_xgb_testing, testing$Species)
result_xgb_training <- caret::confusionMatrix(tab_training)
result_xgb_testing <- caret::confusionMatrix(tab_testing)
result_xgb_testing

# metrics on the training set
sensitivity_xgb_training <- round(result_xgb_training$byClass[, 1]*100, 2)
specificity_xgb_training <- round(result_xgb_training$byClass[, 2]*100, 2)
accuracy_xgb_training <- round(result_xgb_training$overall[1]*100, 2)
precision_xgb_training <- round(mean(result_xgb_training$byClass[,'Pos Pred Value'] *100), 2)
recall_xgb_training <- round(mean(result_xgb_training$byClass[, 'Sensitivity']*100), 2)
f1_score_xgb_training <- round(2 * ((precision_xgb_training * recall_xgb_training) / (precision_xgb_training + recall_xgb_training)),2)

# metrics on the testing set
sensitivity_xgb_testing <- round(result_xgb_testing$byClass[, 1]*100, 2)
specificity_xgb_testing <- round(result_xgb_testing$byClass[, 2]*100, 2)
accuracy_xgb_testing <- round(result_xgb_testing$overall[1]*100, 2)
precision_xgb_testing <- round(mean(result_xgb_testing$byClass[,'Pos Pred Value'] *100), 2)
recall_xgb_testing <-  round(mean(result_xgb_testing$byClass[, 'Sensitivity']*100), 2)
f1_score_xgb_testing <- round(2 * ((precision_xgb_testing * recall_xgb_testing) / (precision_xgb_testing + recall_xgb_testing)),2)


table_metrics <- matrix(c(accuracy_xgb_training, accuracy_xgb_testing,
                          precision_xgb_training, precision_xgb_testing,
                          recall_xgb_training, recall_xgb_testing, 
                          f1_score_xgb_training, f1_score_xgb_testing), byrow = TRUE, ncol = 2)
colnames(table_metrics) <-  c('Training', 'Testing')
rownames(table_metrics) <- c('accuracy', 'precision', 'recall', 'f1-score')
table_metrics

# export the results in LaTex document
print(xtable(table_metrics, type = "latex", digits=2), file = "tables.tex")

# 3.1 plot the confusion matrix
pred_xgb <- table(pred_y_xgb_testing, testing$Species)
tab <- table(pred_xgb_testing, testing$Species)
testing$pred_xgb <- pred_xgb_testing
ggplot(testing, aes(Species, pred_xgb_testing, color = Species)) +
  geom_jitter(wixgbh = 0.2, height = 0.1, size=2) +
  labs(title = 'Confusion Matrix - Decision Tree Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_xgb_testing),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

# alternative
# 3.2 3lot the Confusion Matrix

pred_xgb_testing_species = numeric(length(factor(round(pred_xgb_testing))))
for(i in 1: length(factor(round(pred_xgb_testing)))) {
  if(round(pred_xgb_testing)[i] == 1) {
    pred_xgb_testing_species[i] = 'setosa'
  } else if (round(pred_xgb_testing)[i] == 2) {
    pred_xgb_testing_species[i] = 'versicolor'
  } else {
    pred_xgb_testing_species[i] = 'virginica'
  }
}

cm_xgb <- caret::confusionMatrix(factor(pred_xgb_testing_species), factor(testing$Species), dnn = c("Predicted", "Observed"))
plt <- as.data.frame(cm_xgb$table)
plt$Predicted <- factor(plt$Predicted, levels=rev(levels(plt$Predicted)))

ggplot(plt, aes(Predicted, Observed, fill= Freq)) +
  geom_tile() + 
  geom_text(aes(label=Freq), size = 5) +
  scale_fill_gradient(low="white", high="darkgreen") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c(levels(factor(testing$Species)))) +
  scale_y_discrete(labels=rev(c(levels(factor(testing$Species))))) +
  labs(title = 'Confusion Matrix - XGBoost Classifier',
       subtitle = paste('Predicted vs. Observed from Iris testing dataset. Accuracy: ', accuracy_xgb_testing, '%'),
       y="Predicted", x="Observed") +
  theme(axis.text=element_text(size=8),
        axis.title=element_text(size=8),
        plot.subtitle=element_text(size=10, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))


#----
# end
#----