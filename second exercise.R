full_data <-read.table("dataHW.txt")
head(full_data)
dim(full_data)
names_data <-c("number of car crashes","length of highway","avg daily traffic", "seperation of opposite direction", "width of the lane", "width of the stopping lane")
names(full_data)<-names_data
studentnumber <- 811036
set.seed(studentnumber)
rownumber <- sample(1:nrow(full_data),600, replace = F)
my_data <- full_data[rownumber,]
dim(my_data)
attach(my_data)
library(flexmix)
library(MASS)
library(MuMIn)
library(fic)
library(ggplot2)
library(gridExtra)
#----------------
str(my_data)
levels(my_data$`number of car crashes`)
# factor levels of numb of car crashes are neither increasing nor decreasing
# rearrange the data so that the levels are increasing
my_data$`number of car crashes` <- factor( my_data$`number of car crashes`, c(0,2:19,21, "Y"))
levels(my_data$`number of car crashes`)

P1 <- ggplot(my_data)+
  geom_histogram(aes(`number of car crashes`),stat = "count")

#count data so we think poisson compare poisson distribution with data 
poiss <- rpois(600, 6)
P2 <-ggplot()+
  geom_histogram(aes(poiss), stat = "count")
grid.arrange(P1,P2)
# poiss distr does not capture small and large counts--> overdispersion possible 

# response variable needs to be an integer and other variables need to be numeric

my_data$`number of car crashes` <- as.integer(as.character( my_data$`number of car crashes`))
my_data$`length of highway` <- as.numeric(as.character(my_data$`length of highway`))
my_data$`avg daily traffic` <- as.numeric(as.character(my_data$`avg daily traffic`))
my_data$`seperation of opposite direction`<- as.numeric(as.character(my_data$`seperation of opposite direction`))
my_data$`width of the lane` <- as.numeric(as.character(my_data$`width of the lane`))
my_data$`width of the stopping lane`<- as.numeric(as.character(my_data$`width of the stopping lane`))


fit1 <- glm(my_data$`number of car crashes` ~ ., data = my_data, family = poisson(link = "log"))
summary(fit1) 

fit2 <- glm(`number of car crashes`~., data = my_data, family = quasipoisson(link = "log"))
summary(fit2)
# small overdispersion (1.084986) okay --> use fit1

# giving more oversight to compare different models 
Y <- my_data$`number of car crashes`
X1 <- my_data$`length of highway`
X2 <- my_data$`avg daily traffic`
X3 <- my_data$`seperation of opposite direction`
X4 <- my_data$`width of the lane`
X5 <- my_data$`width of the stopping lane`

fit1.1 <- glm(Y~ (X1+X2+X3+X4+X5)^2, data=my_data, family = poisson(link = "log"))


#exploring best model according to AIC, BIC and Hannan-Quinn cirterion
options(na.action="na.fail")
ms.aic <- dredge(fit1.1, rank = AIC)
ms.bic <- dredge(fit1.1, rank = BIC)
options(na.action = "na.omit")
ms.aic[1:3]
ms.bic[1:3]

hann <- stepAIC(fit1.1, k= log(log(nrow(my_data))), direction =  "both", scope= list(upper=~.^2, lower=~1))
#models chosen: 
# 1 Y ~ X1 + X2 + X3 + X4 + X5 + X1:X4 + X1:X5 + X2:X3 + X2:X5 + X3:X4 + X3:X5 + X4:X5 + X2:X3:X5 + X1:X4:X5
# 2 Y ~ X1 + X2 + X3 + X4 + X5 + X1:X2 + X1:X4 + X1:X5 + X2:X3 + X2:X5 + X3:X4 + X3:X5 + X4:X5 + X2:X3:X5 + X1:X4:X5
# 3 Y ~ X1 + X2 + X3 + X4 + X5 + X1:X4 + X1:X5 + X2:X3 + X2:X4 + X2:X5 + X3:X4 + X3:X5 + X4:X5 + X2:X3:X5 + X1:X4:X5


#creating table for hann-quinn criterion

hann$coefficients
model1 <- c(-4.106909e-01,
            7.304881e+00,
            3.064523e-05,
            -2.979189e-02,
            1.360542e-01,
            6.459752e-01,
            NA,
            -5.872624e-01,
            -1.253742e+00,
            -1.560099e-06,
            NA,
            -1.127963e-06,
            4.161539e-03,
            -2.428646e-03,
            -5.424931e-02,
            1.833821e-07,
            1.063911e-01,
            2.809700e+03)

#getting coefficients for second and third option
second_model_hann_quinn <- glm(Y ~ X1 + X2 + X3 + X4 + X5 + X1:X2 + X1:X4 + X1:X5 + X2:X3 + X2:X5 + X3:X4 + X3:X5 + X4:X5 + X2:X3:X5 + X1:X4:X5, data = my_data, family = poisson(link = "log"))
third_model_hann_quinn  <- glm(Y ~ X1 + X2 + X3 + X4 + X5 + X1:X4 + X1:X5 + X2:X3 + X2:X4 + X2:X5 + X3:X4 + X3:X5 + X4:X5 + X2:X3:X5 + X1:X4:X5, data = my_data, family = poisson(link = "log"))


second_model_hann_quinn$coefficients
model2 <- c(-4.106265e-01,
            7.321315e+00,
            3.107446e-05,
            -3.008368e-02,
            1.356211e-01,
            6.453784e-01,
            -1.121196e-06,
            -5.880768e-01,
            -1.250267e+00,
            -1.551857e-06,
            NA,
            -1.132845e-06 ,
            4.186737e-03,
            -2.435415e-03,
            -5.422344e-02,
            1.829676e-07 ,
            1.062241e-01,
            2811.3 )

third_model_hann_quinn$coefficients
model3 <- c(-1.293766e+00,
            7.631833e+00,
            8.981971e-05, 
            -3.524654e-02,  
            2.100503e-01, 
            6.981175e-01,
            NA,
            -6.147737e-01,
            -1.295779e+00,
            -1.576953e-06,
            -4.930556e-06,
            -1.125916e-06, 
            4.639553e-03,
            -2.456428e-03,
            -5.865532e-02,
             1.852317e-07,
            1.099291e-01,
            2811.3)

hannanquinn_TOP_3 <- rbind.data.frame( model1,model2,model3)

names(hannanquinn_TOP_3) <- c( "(Intercept)" ,
                             "X1",
                             "X2",
                             "X3",
                             "X4",
                             "X5",
                             "X1:X2",
                             "X1:X4", 
                             "X1:X5",
                             "X2:X3",
                             "X2:X4",
                             "X2:X5",
                             "X3:X4",
                             "X3:X5",
                             "X4:X5",
                             "X2:X3:X5",
                             "X1:X4:X5", 
                             "hannan-quinn criterion")


#making tables for AIC and BIC

AIC_TOP_3 <- data.frame(ms.aic$`(Intercept)`[1:3] ,ms.aic$X1[1:3],ms.aic$X2[1:3],ms.aic$X3[1:3],ms.aic$X4[1:3],ms.aic$X5[1:3], ms.aic$`X3:X4`[1:3],ms.aic$AIC[1:3])
BIC_TOP_3 <- data.frame(ms.bic$`(Intercept)`[1:3],ms.bic$X1[1:3],ms.bic$X2[1:3],ms.bic$X3[1:3],ms.bic$X5[1:3],ms.bic$BIC[1:3])

names(AIC_TOP_3) <- c("integer", "x1", "X2","X3", "X4","X5","X3:X4","AIC")
names(BIC_TOP_3) <- c("integer", "X1", "X2","X3","X5","BIC")
# part B
# glm without interactions
fit1.2 <- glm(Y ~ X1+X2+X3+X4+X5, data= my_data, family = poisson(link = "log"))


a <- mean(X1)
b <- mean(X2)
c <- mean(X3)
d <- mean(X4)

width_stopping_lane_8<- c(1,a,b,c,d,8)
width_stopping_lane_10<- c(1,a,b,c,d,10)

X <- rbind("8" = width_stopping_lane_8, 
           "10" = width_stopping_lane_10)

functie <- function(par, X) (X%*%par)
functie(coef(fit1.2), X=X)

inds0 <- c(1,0,0,0,0,0)
comb <- all_inds(fit1.2, inds0)


fic1 <- fic(wide = fit1.2,inds = comb, inds0 =inds0, focus = functie, X=X)
fic1

summary (fic1, adj = TRUE)

fic8 <- data.frame(fic1[1:32,])

fic10 <-data.frame(fic1[33:64,])



