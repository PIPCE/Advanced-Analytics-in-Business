##### without missing factor values #####

#### loading in libraries ####

library(ggplot2)
library(tidyverse)
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(visdat)
library(naniar)
library(simputation)
library(VSURF)
library(zoo)
library(lubridate)
library(Hmisc)
library(ranger)
library(missRanger)
library(pdp)
library(doParallel)

#### loading in datasets ####
train <- read.csv("Assignment 1/korilium/train.csv", header = T, sep = ";", na.strings = c("", " ", "NA"))

dim(train)
str(train)
windows();train %>% vis_miss(warn_large_data = F)

#### set data in the right datatype ####
# function for setting to boolean/logical type
boolean <- function(datastring, negativecondition) {
  for (i in 1:length(datastring)) {
    if (datastring[i] == negativecondition & is.na(datastring[i]) == F) {
      datastring[i] <- FALSE
    }
    if (is.na(datastring[i]) == TRUE) {
      datastring[i] == NA
    }
    else {
      datastring[i] <- TRUE
    }
  }

  datastring <- as.logical(datastring)
  return(datastring)
}
# logical
cols_logical <- c(
  "claim_liable",
  "claim_police",
  "third_party_1_injured",
  "repair_sla",
  "driver_injured"
)
train[cols_logical] <-
  lapply(train[cols_logical], boolean, negativecondition = "N")

# numeric
train$claim_amount <- as.numeric(gsub(",", ".", train$claim_amount))

# factor
cols_factor <- c(
  "fraud", "claim_cause",
  "claim_num_injured",
  "claim_num_third_parties",
  "claim_num_vehicles",
  "claim_alcohol",
  "claim_language",
  "claim_vehicle_brand",
  "claim_vehicle_type",
  "claim_vehicle_fuel_type",
  "policy_holder_form",
  "policy_holder_country",
  "policy_holder_expert_id",
  "driver_form",
  "driver_country",
  "driver_expert_id",
  "third_party_1_vehicle_type",
  "third_party_1_form",
  "third_party_1_country",
  "third_party_1_expert_id",
  "repair_form",
  "repair_country",
  "policy_num_changes",
  "policy_num_claims",
  "policy_coverage_type"
)
train[cols_factor] <- lapply(train[cols_factor], factor)

# date: vehicle_date_inuse
train <- transform(train, claim_vehicle_date_inuse = as.yearmon(as.character(claim_vehicle_date_inuse), "%Y%m"))

# rename factor level of policy_coverage_type

x <- factor(1:73)
levels(train$policy_coverage_type) <- x

#### need to remove columns: ####
# 2nd and 3th party
# dates policy and claim
# id: !!EXCEPT!! for expert and claim id's
# remove the dates, 2de and 3th party  columns

#### ####
train1 <- train %>%
  select(
    -claim_date_registered,
    -claim_date_occured,
    -claim_time_occured,
    -policy_date_start,
    -policy_date_next_expiry,
    -policy_date_last_renewed,
    -(third_party_2_id:third_party_3_expert_id),
    -claim_vehicle_id,
    -policy_holder_id,
    -driver_id,
    -driver_vehicle_id,
    -third_party_1_id,
    -third_party_1_vehicle_id,
    -repair_id
  )
# remove the columns with more than 50% NA
train1.1 <- train1 %>% select(
  -third_party_1_year_birth,
  -repair_year_birth,
  -repair_postal_code,
  -third_party_1_postal_code,
  -policy_holder_expert_id,
  -third_party_1_expert_id,
  -driver_expert_id,
  -repair_form,
  -repair_country,
  -policy_coverage_type, 
  -claim_amount
)

# assume that alcohol is 0 if no value is added
train1.1$claim_alcohol[is.na(train1.1$claim_alcohol)] <- "N"


#### setting year to age for driver, policy holder and vehicle in use ####

train1.1$claim_vehicle_date_inuse <- year(train1$claim_vehicle_date_inuse) + month(train1$claim_vehicle_date_inuse) / 12

train1.1 <- train1.1 %>%
  mutate(policy_holder_age = 2018 - policy_holder_year_birth) %>%
  mutate(driver_age = 2018 - driver_year_birth) %>%
  mutate(claim_vehicle_age = 2018 - claim_vehicle_date_inuse)

#### remove dates and keep ages ####
train1.2 <- train1.1 %>%
  select(
    -policy_holder_year_birth,
    -driver_year_birth,
    -claim_vehicle_date_inuse,
    -ï..claim_id
  )

# one error vehicle age 
train1.2$claim_vehicle_age[train1.2$claim_vehicle_age < 0] <- NA

#### first part of analyses: creating decision trees ####

# without pruning
treemax <- rpart(fraud ~ ., data = train1.2, minsplit = 2, cp = 0)
windows()
plot(treemax)
View(treemax$cptable)
windows()
plotcp(treemax)

#### second part of analyses: Random Forest ####

### cannot deal with missing values use a random forest to impute the missing data

train1.3 <- missRanger(train1.2,
  num.trees = 50,
  max.depth = 10,
  splitrule = "extratrees",
  verbose = T,
  respect.unordered.factors = "order"
)

#### ranger ####
start_time <- Sys.time()
bagging1.1 <- ranger(
  dependent.variable.name = "fraud",
  data = train1.3, num.trees = 100,
  mtry = ncol(train1.3) / 3,
  class.weights = c(1, 1E8),
  probability = F,
  importance = "impurity",
  verbose = T
  )
end_time <- Sys.time()

end_time - start_time

#### third part of analyses: studying partial effects ####


#### Variable importance ####

windows()
par(mar = c(4, 11, 4, 4))
barplot(bagging1.1$variable.importance,
  horiz = T,
  las = 1,
  cex.names = 0.8,
  cex.axis = 1,
  beside = F,
  col = "#5659ddfa"
)


####  variable selection (cload computing)

x <- train1.3[, -1]
y <- train1.3[, 1]
cl <- makeCluster(4, type = "PSOCK")
registerDoParallel(cl)
test <- VSURF(x, y,
  ntree = 10,
  nfor.thres = 10,
  nfor.pred = 8,
  nfor.interp = 8,
  mptry <- (ncol(train1.3) - 1) / 3,
  parallel = TRUE,
  ncores = detectCores(),
  RFimplem = "ranger"
)
summary(test)
colnames(train1.3[, (test$varselect.thres + 1)])
windows()
plot(test)
plot(test, step = "thres", imp.mean = FALSE, ylim = c(0, 2e-4))


stopCluster(cl)
# use parallel backend to optimize computation
cl <- makeCluster(4)
registerDoParallel(cl)
windows()
partial(bagging1.1, pred.var = "claim_amount", plot = TRUE)

bagging1.1 %>%
  partial(pred.var = "claim_amount") %>%
  plotPartial(smooth = TRUE, lwd = 2, ylab = expression(f(claim_vehicle_age)))

pd <- partial(bagging1.1,
  pred.var = c("claim_amount", "policy_premium_100"),
  grid.resolution = 15, chull = TRUE, parallel = TRUE, paropts = list(.packages = "ranger")
)

windows()
plotPartial(pd)
windows()
plotPartial(pd,
  levelplot = FALSE,
  zlab = "fraud",
  drape = TRUE,
  colorkey = TRUE,
  screen = list(z = -20, x = -60)
)

stopCluster(cl)
# ice graph # not so good

pred.ice <- function(object, newdata) predict(object, newdata)$predictions

start_time <- Sys.time()
windows()
partial(bagging1.1,
  pred.var = "claim_vehicle_age",
  ice = TRUE,
  plot = TRUE, alpha = 0.2,
  parallel = TRUE, paropts = list(.packages = "ranger"),
  grid.resolution = 15
)
end_time <- Sys.time()

end_time - start_time





#### To Do ####
# overweighting the fraud cases based on the claim
# taking into account NA as information
# tuning => tuningranger package
# variable selection and importance