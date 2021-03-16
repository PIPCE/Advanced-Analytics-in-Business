library(rpart)
library(randomForest)
library(VSURF)

 data("spam", package = "kernlab")
 data("Ozone", package = "mlbench")
 data("vac18", package = "mixOmics")
 data("jus", package = "VSURF")
 set.seed(9146301)
 levels(spam$type) <- c("ok", "spam")
 yTable <- table(spam$type)
 indApp <- c(sample(1:yTable[2], yTable[2]/2),
sample((yTable[2] + 1):nrow(spam), yTable[1]/2))
 spamApp <- spam[indApp, ]
 spamTest <- spam[-indApp, ]
