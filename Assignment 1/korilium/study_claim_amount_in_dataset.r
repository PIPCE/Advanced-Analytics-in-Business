

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
                  header = T, sep = ";", na.strings = c("", " ", "NA"))

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





# use parallel backend to optimize computation
cl <- makeCluster(4)
registerDoParallel(cl)
windows()
partial(bagging1.1, pred.var = "claim_amount", plot = TRUE)

pd <- partial(bagging1.1,
  pred.var = c("claim_amount", "policy_premium_100"),
  grid.resolution = 15, chull = TRUE,
  parallel = TRUE, paropts = list(.packages = "ranger")
)

pd1 <- partial(bagging1.1,
        pred.var = c("claim_amount", "policy_coverage_1000"),
        grid.resolution = 15, chull = TRUE, parallel = TRUE, 
        paropts = list(.packages = "ranger"))


windows();
plotPartial(pd,
  levelplot = FALSE,
  zlab = "fraud",
  drape = TRUE,
  colorkey = TRUE,
  screen = list(z = -20, x = -60)
)
windows();
plotPartial(pd1,
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
