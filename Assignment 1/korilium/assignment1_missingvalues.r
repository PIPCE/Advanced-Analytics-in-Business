##### with missing factor values #####
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
library(lattice)
library(VIM)
library(mice)
library(tuneRanger)

#### loading in datasets ####
train <- read.csv("Assignment 1/korilium/train.csv",
  header = T, sep = ";", na.strings = c("", " ", "NA")
)

dim(train)
str(train)
# windows();train %>% vis_miss(warn_large_data = F)

#### set data in the right datatype ####
# function for setting to boolean/logical type
boolean <- function(datastring, negativecondition) {
  for (i in 1:length(datastring)) {
    if (datastring[i] == negativecondition & is.na(datastring[i]) == F) {
      datastring[i] <- FALSE
    }
    if (is.na(datastring[i]) == TRUE) {
      datastring[i] <- NA
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
  "policy_coverage_type",
  "repair_id",
  "third_party_3_expert_id",
  "third_party_3_country",
  "third_party_3_form",
  "third_party_3_vehicle_type",
  "third_party_3_injured",
  "third_party_3_id",
  "third_party_2_expert_id",
  "third_party_2_country",
  "third_party_2_form",
  "third_party_2_vehicle_type",
  "third_party_2_injured",
  "third_party_2_id",
  "third_party_1_id",
  "third_party_1_injured"
)
train[cols_factor] <- lapply(train[cols_factor], factor, exclude = NULL)

# date: vehicle_date_inuse
train <- transform(train, claim_vehicle_date_inuse = as.yearmon(as.character(claim_vehicle_date_inuse), "%Y%m"))

# rename factor level of policy_coverage_type

x <- factor(1:73)
levels(train$policy_coverage_type) <- x

#### need to remove columns: ####
# dates policy and claim
# id: policy holder, claim_vehicle 1-3, driver, repair_id, driver_vehicle, third_party_1-3 and claim id
# claim amount

#### ####
train1 <- train %>%
  select(
    -policy_date_start,
    -policy_date_next_expiry,
    -policy_date_last_renewed,
    -claim_time_occured,
    -claim_date_registered,
    -claim_date_occured,
    -driver_id,
    -policy_holder_id,
    -claim_vehicle_id,
    -ï..claim_id,
    -driver_vehicle_id,
    -third_party_3_vehicle_id,
    -third_party_2_vehicle_id,
    -third_party_1_vehicle_id,
    -third_party_1_id,
    -third_party_2_id,
    -third_party_3_id,
    -repair_id,
    -claim_amount
  )

#  windows();train1 %>% vis_miss(warn_large_data = F)


#### setting year to age for:
# repair_year_birth,
# driver_year_birth,
# policy_holder_year_birth,
# claim_vehicle_date_inuse,
# third_party_3_year_birth
# third_party_2_year_birth
# third_party_1_year_birth

train1$claim_vehicle_date_inuse <- year(train1$claim_vehicle_date_inuse) + month(train1$claim_vehicle_date_inuse) / 12

train1.1 <- train1 %>%
  mutate(policy_holder_age = 2017 - policy_holder_year_birth) %>%
  mutate(driver_age = 2017 - driver_year_birth) %>%
  mutate(claim_vehicle_age = 2017 - claim_vehicle_date_inuse) %>%
  mutate(repair_shop_age = 2017 - repair_year_birth) %>%
  mutate(third_party_1_age = 2017 - third_party_1_year_birth) %>%
  mutate(third_party_2_age = 2017 - third_party_2_year_birth) %>%
  mutate(third_party_3_age = 2017 - third_party_3_year_birth)

train1.1$claim_alcohol[is.na(train1.1$claim_alcohol)] <- "N"

#### remove dates and keep ages ####
train1.2 <- train1.1 %>%
  select(
    -policy_holder_year_birth,
    -driver_year_birth,
    -claim_vehicle_date_inuse,
    -repair_year_birth,
    -third_party_1_year_birth,
    -third_party_2_year_birth,
    -third_party_3_year_birth,
  )

# one error vehicle age
train1.2$claim_vehicle_age[train1.2$claim_vehicle_age < 0] <- NA

windows()
marginplot(train1.2[, c("policy_holder_age", "driver_age")],
  col = mdc(1:2), cex.numbers = 1.2, pch = 19
)
#### look at the ages ####

windows()
ggplot(train1.2) +
  geom_density(aes(policy_holder_age, colour = "policy holder age")) +
  geom_density(aes(driver_age, colour = "driver age")) +
  geom_density(aes(claim_vehicle_age, colour = "claim vehicle age")) +
  scale_colour_manual("",
    breaks = c("policy holder age", "driver age", "claim vehicle age"),
    values = c("red", "blue", "green")
  )
windows()
ggplot(train1.2) +
  geom_density(aes(repair_shop_age, colour = "repair shop age")) +
  geom_density(aes(third_party_1_age, colour = "third party 1 age")) +
  geom_density(aes(third_party_2_age, colour = "third party 2 age")) +
  scale_color_manual("",
    breaks = c(
      "repair shop age", "third party 1 age",
      "third party 2 age"
    ),
    values = c("orange", "green", "blue")
  )

# look at postal code
windows()
ggplot(train1.2) +
  geom_density(aes(third_party_1_postal_code, colour = "third_party_1_postal_code")) +
  geom_density(aes(third_party_2_postal_code, colour = "third_party_2_postal_code")) +
  geom_density(aes(third_party_3_postal_code, colour = "third_party_3_postal_code")) +
  geom_density(aes(repair_postal_code, colour = "repair_postal_code")) +
  geom_density(aes(driver_postal_code, colour = "driver_postal_code")) +
  geom_density(aes(claim_postal_code, colour = "claim_postal_code")) +
  geom_density(aes(policy_holder_postal_code, colour = "policy_holder_postal_code")) +
  scale_colour_manual("",
    breaks = c(
      "third_party_1_postal_code", "third_party_2_postal_code", "third_party_3_postal_code",
      "repair_postal_code", "driver_postal_code", "claim_postal_code", "policy_holder_postal_code"
    ),
    values = c("red", "blue", "green", "grey", "black", "orange", "forestgreen")
  )


#### create factor levels for all missing values for age and postal code ####

ages <- c(
  "repair_shop_age", "policy_holder_age", "third_party_1_age",
  "third_party_2_age", "driver_age", "claim_vehicle_age",
  "third_party_3_age", "third_party_1_postal_code",
  "third_party_3_postal_code", "third_party_2_postal_code",
  "repair_postal_code", "driver_postal_code", "claim_postal_code",
  "policy_holder_postal_code"
)

for (i in 1:length(ages)) {
  if (class(unlist(train1.2[ages[i]])) == "numeric" | class(unlist(train1.2[ages[i]])) == "integer") {
    train1.2[ages[i]] <-
      factor(cut(as.numeric(unlist(train1.2[ages[i]])), breaks = 10), exclude = "")
  }
}




windows()
vis_miss(train1.2, warn_large_data = F)
windows()
vis_dat(train1.2, warn_large_data = F)
windows()
gg_miss_upset(train1.2, nsets = 16, nintersects = 16)

#### look at missing data
train1.2$claim_vehicle_load[train1.2$claim_vehicle_load > 300] <- NA
windows()
marginplot(train1.2[, c("claim_vehicle_load", "claim_vehicle_power")],
  col = mdc(1:2), cex.numbers = 1.2, pch = 19
)
windows()
marginplot(train1.2[, c("claim_vehicle_cyl", "claim_vehicle_power")],
  col = mdc(1:2), cex.numbers = 1.2, pch = 19
)

#### first part of analyses: creating decision trees ####

# without pruning
treemax <- rpart(fraud ~ ., data = train1.2, minsplit = 4, cp = 0)
windows()
plot(treemax)
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

bagging1.1 <- ranger(
  dependent.variable.name = "fraud",
  data = train1.3, num.trees = 100,
  mtry = ncol(train1.3) / 3,
  class.weights = c(1, 1E8),
  probability = TRUE,
  importance = "impurity",
  verbose = T
)



####  variable selection (cload computing) #CPU to slow!!!!

x <- train1.3[, -1]
y <- train1.3[, 1]
cl <- makeCluster(4, type = "PSOCK")
registerDoParallel(cl)
test <- VSURF(x, y,
  ntree = 250,
  nfor.thres = 50,
  nfor.pred = 25,
  nfor.interp = 25,
  mptry <- (ncol(train1.3) - 1) / 3,
  parallel = TRUE,
  ncores = detectCores(),
  RFimplem = "ranger",
  max.depth = 60,
  num.random.split = 1
)

test1.1 <- VSURF_thres(x, y,
  ntree = 300, RFimplem = "ranger",
  parallel = TRUE
)

VSURF_interp(x, y,
  ntree = 50, RFimplem = "ranger",
  parallel = TRUE, vars = test1.1$varselect.thres,
  max.depth = 40
)

VSURF_pred(x, y,
  ntree = 300, RFimplem = "ranger",
  parallel = TRUE
)
summary(test)
colnames(train1.3[, (test$varselect.thres + 1)])

colnames(train1.3[, (test$varselect - +1)])

windows()
plot(test)
windows()
plot(test, step = "thres", imp.mean = FALSE, ylim = c(0, 2e-4))

# using variable selection and then tune ranger
train1.4 <- train1.3[, c(1, test$varselect.thres + 1)]

bagging1.2 <- ranger(
  dependent.variable.name = "fraud",
  data = train1.4, num.trees = 100,
  mtry = (ncol(train1.3) / 4),
  sample.fraction = 0.5,
  class.weight = c(1e-5, 1e5),
  probability = F,
  importance = "impurity",
  verbose = T,
  classification = T
)
#### CPU too slow!!!
bagging.task <- makeClassifTask(data = train1.4, target = "fraud")

# estimate Tuning
estimateTimeTuneRanger(bagging.task, num.trees = 100, num.threads = 2, iter = 70)
# Tuning

res <- tuneRanger(bagging.task, measure = list(auc))






#### Variable importance ####

windows()
par(mar = c(4, 7, 4, 4), mfrow = c(1, 2))

barplot(bagging1.1$variable.importance[1:26],
  horiz = T,
  las = 1,
  cex.names = 0.6,
  cex.axis = 1,
  beside = F,
  col = "#5659ddfa"
)
barplot(bagging1.1$variable.importance[27:58],
  horiz = T,
  las = 1,
  cex.names = 0.6,
  cex.axis = 1,
  beside = F,
  col = "#5659ddfa"
)

windows()
par(mar = c(4, 7, 4, 4))
barplot(test$imp.varselect.thres,
  horiz = T,
  las = 1,
  cex.names = 0.6,
  beside = F,
  col = "red",
  names.arg = colnames(train1.3[, (test$varselect.thres + 1)])
)




#### To Do ####
# overweighting the fraud cases based on the claim
# tuning => tuningranger package
# variable selection and importance