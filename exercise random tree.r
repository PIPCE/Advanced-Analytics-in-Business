library(tidyverse)
library(ISLR)
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)

detach(train1)
attach(Carseats)
High <- ifelse(Sales<8, "No","Yes")
Carseats <- data.frame(Carseats,High)
########################## decision tree for depiction ############
d_tree <- rpart(High ~ .-Sales, Carseats)
rpart.plot(d_tree, main="Full Data Set Decision Tree", fallen.leaves=FALSE, extra=104, box.palette="GnBu")


