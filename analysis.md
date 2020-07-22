---
title: "Analysis"
author: "Shivam_Rajoriya"
date: "7/22/2020"
output: html_document
---
loading packages
```{r}
library(caret)
library(rpart)
library(rpart.plot)
library(knitr)
library(corrplot)
library(randomForest)
library(rattle)
library(RColorBrewer)
```
data loading and cleaning 
```{r}
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]

dim(TrainSet)
```

```{r}

dim(TestSet)
```
so above we could see there are 160 variables and in the cleaning process we would remove near zero variance variables ,id variables and NA variables.
```{r}
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
dim(TrainSet)
```

```{r}
dim(TestSet)
```

```{r}

AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
dim(TrainSet)
```

```{r}
dim(TestSet)

```

```{r}

TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
```

```{r}
dim(TestSet)
```

now after the cleaning process we are left only with 54 variables.
ANALYSIS:-
first lets see the correlation between different variables
```{r}
corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

The highly correlated variables are shown in dark colors in the graph above. To make an evem more compact analysis, a PCA (Principal Components Analysis) could be performed as pre-processing step to the datasets. Nevertheless, as the correlations are quite few, this step will not be applied for this assignment.

predictive modelling:-
method1:- Decision trees
```{r}
set.seed(12345)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDecTree)
```
predictions
```{r}
predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)
confMatDecTree
```

```{r}
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
```
method2:- generalised boosted method

```{r}
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
```

```{r}
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM
```

```{r}
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))
```

method3:-
random forest
```{r}
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel
```

```{r}
predictRandForest <- predict(modFitRandForest, newdata=TestSet)
confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)
confMatRandForest
```

```{r}
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))
```

FINAL STEP:-
The accuracy of the 3 regression modeling methods above are:

Random Forest : 0.9963
Decision Tree : 0.7368
GBM : 0.9839
In that case, the Random Forest model will be applied to predict the 20 quiz results (testing dataset) as shown below.

```{r}
predictTEST <- predict(modFitRandForest, newdata=testing)
predictTEST
```
