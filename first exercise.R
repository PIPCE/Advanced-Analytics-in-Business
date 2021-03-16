
O <- seq(0.1,1.5,0.1)
Y <- seq(0.1,1.5,0.1)

calc<- function(y,Y,O){
  (Y*y*exp(-Y*y)/y)*(log(Y*y*exp(-Y*y)/y)-log(O*exp(-O*y)))
}

parameter_values <- setNames(data.frame(matrix(ncol=15,nrow=15)),
                             c("Y=0.1",
                               "Y=0.2",
                               "Y=0.3",
                               "Y=0.4",
                               "Y=0.5",
                               "Y=0.6",
                               "Y=0.7", 
                               "Y=0.8",
                               "Y=0.9",
                               "Y=1",
                               "Y=1.1",
                               "Y=1.2",
                               "Y=1.3",
                               "Y=1.4", 
                               "Y=1.5"))
                                        

for ( i in 1:length(Y)){ 
  for(j in 1:length(O)) {
    parameter_values[i,j] <- integrate(calc, lower = 0, upper = 100,Y[i], O[j])$value
    }}

library(ggplot2)
library(gridExtra)

p1<-ggplot(parameter_values)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.1`, color = "??=0.1"),size=1)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.2`, color = "??=0.2"),size=1)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.3`, color = "??=0.3"), size=1)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.4`, color=  "??=0.4"), size=1)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.5`, color=  "??=0.5"), size=1)+
  geom_hline(aes(yintercept = 0))+
  labs(x="value for ??", y="Kullback-Leibler distance")
  

p2 <-ggplot(parameter_values)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.6`, color="??=0.6"),size=1)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.7`, color="??=0.7"), size=1)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.8`, color="??=0.8"), size=1)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=0.9`, color="??=0.9"), size=1)+
  geom_line(aes(x =seq(0.1,1.5,0.1), parameter_values$`Y=1`, color = "??=1"), size=1)+
  geom_hline(aes(yintercept = 0))+
  labs(x="value for ??", y="Kullback-Leibler distance")

grid.arrange(p1,p2)




