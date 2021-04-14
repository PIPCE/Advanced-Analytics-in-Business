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
library(Information)
library(gridExtra)
library(binaryLogic)
library(lubridate)
library(dplyr)
library(httpgd)
library(ClustOfVar)
library(knitr)
library(plyr)
library(PCAmixdata)
library(reshape2)
 library(ggnewscale)
hgd()
hgd_browse()

train <- read.csv("train.csv",
  header = T, sep = ";", na.strings = c("", " ", "NA")
)

dim(train)
str(train)


#### set data in the right datatype ####
# function for setting to boolean/logical type
boolean <- function(datastring, negativecondition) {
  for (i in 1:length(datastring)) {
    if (is.na(datastring[i]) == TRUE) {
      datastring[i] <- NA
    }
    if (datastring[i] == negativecondition & is.na(datastring[i]) == FALSE ) {
      datastring[i] <- 0
    }
    else {
      datastring[i] <- 1
    }
  }
  datastring <- as.factor(datastring)
  return(datastring)
}
# logical
cols_logical <- c(
    "fraud",
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
  "claim_cause",
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
train[cols_factor] <- lapply(train[cols_factor], function(x) factor(x ,exclude = NULL, levels = unique(x) ))

# rename factor level of policy_coverage_type

x <- factor(1:73)
levels(train$policy_coverage_type) <- x 


#  windows();train1 %>% vis_miss(warn_large_data = F)


#### setting integer to numeric and setting year to age for:
# repair_year_birth,
# driver_year_birth,
# policy_holder_year_birth,
# claim_vehicle_date_inuse,
# third_party_3_year_birth
# third_party_2_year_birth
# third_party_1_year_birth

# date: vehicle_date_inuse
train <- transform(train, claim_vehicle_date_inuse = as.yearmon(as.character(claim_vehicle_date_inuse), "%Y%m"))
train$claim_vehicle_date_inuse <- year(train$claim_vehicle_date_inuse) + month(train$claim_vehicle_date_inuse) / 12

train1<- train %>%
  mutate(policy_holder_age = 20170000 - policy_holder_year_birth*10000) %>%
  mutate(driver_age = 20170000 - driver_year_birth*10000) %>%
  mutate(claim_vehicle_age = 20170000 - claim_vehicle_date_inuse) %>%
  mutate(repair_shop_age = 20170000 - repair_year_birth*10000) %>%
  mutate(third_party_1_age = 20170000 - third_party_1_year_birth*10000) %>%
  mutate(third_party_2_age = 20170000 - third_party_2_year_birth*10000) %>%
  mutate(third_party_3_age = 20170000 - third_party_3_year_birth*10000)%>%
  mutate(claim_age = 20170000 - claim_date_occured)%>% 
  mutate(policy_start_age = 20170000 - policy_date_start*100)%>% 
  mutate(policy_next_expiry_age = 20170000 - policy_date_next_expiry*100)





train1$claim_alcohol[is.na(train1$claim_alcohol)] <- "N"


#### need to remove columns: ####
# dates
# id: policy holder, claim_vehicle 1-3, driver, repair_id, driver_vehicle, claim_vehicle_id and claim id
# claim amount
# policy date last renewed (it is the same as date next expiry)
# only  postal code  of claim needed see plot 

#### ####
train1.2 <- train1 %>%
  select(
    -policy_holder_year_birth,
    -driver_year_birth,
    -claim_vehicle_date_inuse,
    -repair_year_birth,
    -third_party_1_year_birth,
    -third_party_2_year_birth,
    -third_party_3_year_birth,
    -claim_time_occured, 
    -claim_date_occured,
    -claim_date_registered, 
    -policy_date_start, 
    -policy_date_next_expiry, 
    -claim_amount, 
    -policy_date_start,
    -policy_date_next_expiry,
    -policy_date_last_renewed,
    -claim_time_occured,
    -claim_date_registered,
    -claim_date_occured,
    -driver_id,
    -policy_holder_id,
    -claim_vehicle_id,
    -driver_vehicle_id,
    -claim_vehicle_id, 
    -third_party_3_vehicle_id,
    -third_party_2_vehicle_id,
    -third_party_1_vehicle_id,
    -third_party_1_id,
    -third_party_2_id,
    -third_party_3_id,
    -policy_date_last_renewed,
    -repair_id,
    -claim_amount, 
    -ï..claim_id, 

    )

# one error vehicle age
train1.2$claim_vehicle_age[train1.2$claim_vehicle_age < 0] <- NA



str(train1.2)

# supervised binning 
# variables that need to be binned: 


test <- splitmix(train1.2)
numeric_data <- train1.2[,test$col.quant]
non_numeric_data <- train1.2[,test$col.qual]

train1.2.quanti <- data.frame(apply(numeric_data,2, as.numeric))
train1.2.quali <- data.frame(non_numeric_data)
train1.2.quali <- train1.2.quali %>% 
select( - third_party_3_expert_id)# all NA


tree <- hclustvar(X.quanti = train1.2.quanti,X.quali=train1.2.quali)
plot(tree)
nvars <- length(tree[tree$height<1])
part_inti <- cutreevar(tree, nvars, matsim=T)
part_inti$var
part_inti$sim
part_inti$cluster
part_inti$E

#kmeans <-kmeansvar(X.quanti= train1.2.quanti,
#              X.quali = train1.2.quali, init= part_inti$cluster)
# dependent factor needs to be numeric 

train1.2$fraud <- as.numeric(train1.2$fraud)

for( i in 1:length(train1.2$fraud)){
  if(train1.2$fraud[i] == 1 ){
    train1.2$fraud[i] = 0 
  }else{
    train1.2$fraud[i] = 1
  }
}


IV <- Information::create_infotables(data=train1.2, y="fraud")
names <- names(IV$Tables)

kable(IV$Summary)

IV$Tables$driver_expert_id$WOE

plot_infotables(IV, "claim_cause")
MultiPlot(IV, names)

kable(IV$Tables)


clusters <- cbind.data.frame(melt(part_inti$cluster), row.names(melt(part_inti$cluster)))

loadings = data.frame()
for( i in 1:30){
  loadingcluster <- data.frame(x = part_inti$var[i])[1]
 colnames(loadingcluster) <- c("loadings")
  loadings <- rbind(loadings, loadingcluster)
}
setDT(loadings, keep.rownames =TRUE)[]

names(clusters) <- c("Cluster", "Variable")
colnames(loadings) <- c("Variable", "loadings")


clusters <- join(clusters, IV$Summary, by= "Variable", type="left")
clusters <- join(clusters, loadings, by= "Variable", type="left")
clusters <- clusters[order(clusters$Cluster),]
clusters

count_clusters <- count(clusters, 1)
plot_choosing_cluster <-  subset(clusters, Cluster %in% count_clusters$Cluster[count_clusters$freq!= 1])
clusters$Cluster <- as.factor(clusters$Cluster)
plot_choosing_cluster$Cluster <- as.factor(plot_choosing_cluster$Cluster)

ggplot(clusters, aes(reorder(Variable,IV), IV))+
  geom_col(aes(fill = Cluster))+
  coord_flip()
#