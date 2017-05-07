setwd("C:\\Users\\ankitswarnkar\\Projects\\Data Mining Project")
library(readxl)
library(MASS)
library(ROSE) # imbalance data
#Trend Line Graph#
data=read_excel("default of credit card clients.xls")
#Q. Does defaulter have any relation with month
data_desc = data[1,]
##Conver the data to Data matix
#Remove non-numeric
data = data[-1,]

#data.rose <- ROSE(Y ~ ., data = data, seed = 1)$data

##DATA EXPLORATION
datamat = data.matrix(data)
hist(datamat[,"Y"],freq = TRUE, main = "Distribution of Data", xlab = "Defaulter VS Non-Defaulter",ylab = "Count")
defaulter_data = datamat[datamat[,"Y"]==1,]
nondefaulter_data = datamat[datamat[,"Y"]==0,]
#Wheter there is any trend in defaulter and month
tempd =  defaulter_data[,7:12]
t1<-rep(0,5)
for (i in 7:12){
  val = length(defaulter_data[defaulter_data[,i]>0,i])
  t1[i-7] = val
}
t2<-rep(0,5)
for (i in 7:12){
  val = length(nondefaulter_data[nondefaulter_data[,i]>0,i])
  t2[i-7] = val
}
plot(t2,type = "l",xlab = "Months", main = "Month wise plot of frequency of defauters", col= "blue",ylab = "Counts of defaulting case")
lines(t1,type = "l", col= "red")

#Does married guys have more chance of being defaulter?

t1<-rep(0,3) # to hold values
total_men <- length(defaulter_data[defaulter_data[,3]==1,])
t1[1] <- length(defaulter_data[defaulter_data[,5]==1 & defaulter_data[,3]==1,])  #Married Men
t1[2] <- length(defaulter_data[defaulter_data[,5]==2 & defaulter_data[,3]==1,])  #Single Men
t1[3] <- length(defaulter_data[(defaulter_data[,5]==3| defaulter_data[,5]==0 )& defaulter_data[,3]==1,]) #Others
colors = c("blue", "yellow", "green")
labels_pie = c("Married", "Single", "Others")
pie(t1,col=colors, labels = labels_pie)
#Does married Women have more chance of being defaulter?
t1<-rep(0,3) # to hold values
total_men <- length(defaulter_data[defaulter_data[,3]==2,])
t1[1] <- length(defaulter_data[defaulter_data[,5]==1 & defaulter_data[,3]==2,])  #Married WoMen
t1[2] <- length(defaulter_data[defaulter_data[,5]==2 & defaulter_data[,3]==2,])  #Single WoMen
t1[3] <- length(defaulter_data[(defaulter_data[,5]==3| defaulter_data[,5]==0 )& defaulter_data[,3]==2,]) #Others
colors = c("red", "green", "cyan")
labels_pie = c("Married", "Single", "Others")
pie(t1,col=colors, labels = labels_pie)

#pair plots
png(file = "scatterplot_matrices.png")
pairs(~X2+X3+X4+X5+Y,data = datamat,main = "Scatterplot Matrix")

##Over Sampling 
#Over Estimation Techniques
data <- ovun.sample(Y ~ ., data = data, method = "over", N = 50000)$data
#datamat = data.matrix(data)

####Classification starts from next file####