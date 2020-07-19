#importing data file
data=read.csv("C:/Users/hp/Desktop/hackathon/train (1).csv")
names(data)
data <- data[-c(1236),] 
str(data)
input<- data[,c("subscriber","Trend_day_count","Tag_count","comment_count","Trend_tag_count","likes","dislike","views")]
names(input)

#converting variables from object to integer
#1
input$subscriber=as.integer(input$subscriber)
is.integer(input$subscriber)
#2
input$Trend_day_count=as.integer(input$Trend_day_count)
is.integer(input$Trend_day_count)
#3
input$Tag_count=as.integer(input$Tag_count)
is.integer(input$Tag_count)
#4
input$comment_count=as.integer(input$comment_count)
is.integer(input$comment_count)
#5
input$Trend_tag_count=as.integer(input$Trend_tag_count)
is.integer(input$Trend_tag_count)
#6
input$likes=as.integer(input$likes)
is.integer(input$likes)
#7
input$dislike=as.integer(input$dislike)
is.integer(input$dislike)
#8
input$views=as.integer(input$views)
is.integer(input$views)

str(input)

#identifying missing values
sapply(input,function(x) sum(is.na(x)))

#replacing missing values with mean 
input$likes[is.na(input$likes)] <- round(mean(input$likes, na.rm = TRUE))
sapply(input,function(x) sum(is.na(x)))

## replacing missing values with mean 
input$comment_count[is.na(input$comment_count)] <- round(mean(input$comment_count, na.rm = TRUE))
sapply(input,function(x) sum(is.na(x)))

## replacing missing values with mean 
input$subscriber[is.na(input$subscriber)] <- round(mean(input$subscriber, na.rm = TRUE))
sapply(input,function(x) sum(is.na(x)))

## replacing missing values with mean 
input$comment_count[is.na(input$comment_count)] <- round(mean(input$comment_count, na.rm = TRUE))
sapply(input,function(x) sum(is.na(x)))

## replacing missing values with mean 
input$Trend_day_count[is.na(input$Trend_day_count)] <- round(mean(input$Trend_day_count, na.rm = TRUE))
sapply(input,function(x) sum(is.na(x)))

##checking outliers and treating them
#1
boxplot(input$subscriber)
summary(input$subscriber)

#creating upper limit value
upper<-3.825e+06+1.5*IQR(input$subscriber)
upper

#creating lower limit value
lower<-2.429e+05-1.5*IQR(input$subscriber)
lower

# upper limit replacement
input$subscriber[input$subscriber > upper]<-upper
summary(input$subscriber)
boxplot(input$subscriber)

# lower limit replacement
input$subscribe[input$subscribe < lower]<-lower
summary(input$subscriber)

#2
boxplot(input$Trend_day_count)
summary(input$Trend_day_count)

#creating upper limit value
upper<-10.000+1.5*IQR(input$Trend_day_count)
upper

#creating lower limit value
lower<-4.000-1.5*IQR(input$Trend_day_count)
lower

# upper limit replacement
input$Trend_day_count[input$Trend_day_count > upper]<-upper
summary(input$Trend_day_count)
boxplot(input$Trend_day_count)

# lower limit replacement
input$Trend_day_count[input$Trend_day_count < lower]<-lower
summary(input$Trend_day_count)

#3
boxplot(input$Tag_count)

#4
boxplot(input$comment_count)
summary(input$comment_count)

#creating upper limit value
upper<-203240+1.5*IQR(input$comment_count)
upper

#creating lower limit value
lower<-126760-1.5*IQR(input$comment_count)
lower

# upper limit replacement
input$comment_count[input$comment_count > upper]<-upper
summary(input$comment_count)
boxplot(input$comment_count)

# lower limit replacement
input$comment_count[input$comment_count < lower]<-lower
summary(input$comment_count)

#5
boxplot(input$Trend_tag_count)

#6
boxplot(input$likes)
summary(input$likes)

#creating upper limit value
upper<-10.000+1.5*IQR(input$likes)
upper

#creating lower limit value
lower<-4.000-1.5*IQR(input$likes)
lower

# upper limit replacement
input$likes[input$likes > upper]<-upper
summary(input$likes)
boxplot(input$likes)

# lower limit replacement
input$likes[input$likes < lower]<-lower
summary(input$likes)

#7
boxplot(input$dislike)

#8
boxplot(input$views)

#correlation
plot(input)

par(mfrow=c(2,2))
plot(views~.,data=input)

cor(input)


# Correlation Matrix & plot(Scatter plot) , co-linearity & multi colinearity
attach(input)
cor(input)
plot(views,likes)
plot(views,dislike)
plot(views,Trend_tag_count)
plot(views,comment_count)
plot(views,Tag_count)
plot(views,Trend_day_count)
plot(views,subscriber)

# Model Building
##Multiple Logistic Regression
set.seed(12)
library(caret)
Train<-createDataPartition(input$views,p=0.7,list=FALSE)
training<-input[Train,]
testing<-input[-Train,]

#Enter method
# Model Creation -we reject Ho (pvalue<alpha){we acceptH1}
model<-lm(views~subscriber+Trend_day_count+Tag_count+comment_count+Trend_tag_count+likes+dislike,data = training)
summary(model)

#variance inflation factor
library(car)
vif(model)

# forward method
model1<-step(lm(views~.,data=training),
             direction="forward")
summary(model1)
library(car)
vif(model1)

# backward method
model2<-step(lm(views~.,data = training)
             ,direction = "backward")
summary(model2)
library(car)
vif(model2)

# both method
model2<-step(lm(views~.,data = training)
             ,direction = "both")
summary(model2)
library(car)
vif(model2)
exp(coef(model2))

#Adjusted r square - better model as greater than 70%

# assumption
par(mfrow=c(2,2))
plot(model2)
library(lmtest)
dwtest(model2)
ncvTest(model2)

##JUST TO CHECK MATHEMATICALLY of linear Model
training$Fitted_value<-model2$fitted.values
training$Residual<-model2$residuals
sum(training$Residual)


##Prediction on test data
testing$Predicted<-Predict(model2,testing)
testing$Residual<-testing$views-testing$Predicted
sum(testing$Residual)


################################### TEST DATA ########################################

#importing data file
data1=read.csv("C:/Users/hp/Desktop/hackathon/test (1).csv")
names(data1)
data1 <- data1[-c(1236),] 
str(data1)
input1<- data1[,c("subscriber","Trend_day_count","Tag_count","comment_count","Trend_tag_count","likes","dislike")]
names(input1)

#converting variables from object to integer
#1
input1$subscriber=as.integer(input1$subscriber)
is.integer(input1$subscriber)
#2
input1$Trend_day_count=as.integer(input1$Trend_day_count)
is.integer(input1$Trend_day_count)
#3
input1$Tag_count=as.integer(input1$Tag_count)
is.integer(input1$Tag_count)
#4
input1$comment_count=as.integer(input1$comment_count)
is.integer(input1$comment_count)
#5
input1$Trend_tag_count=as.integer(input1$Trend_tag_count)
is.integer(input1$Trend_tag_count)
#6
input1$likes=as.integer(input1$likes)
is.integer(input1$likes)
#7
input1$dislike=as.integer(input1$dislike)
is.integer(input1$dislike)

str(input1)

#identifying missing values
sapply(input1,function(x) sum(is.na(x)))

#replacing missing values with mean 
input1$likes[is.na(input1$likes)] <- round(mean(input1$likes, na.rm = TRUE))
sapply(input1,function(x) sum(is.na(x)))

## replacing missing values with mean 
input1$comment_count[is.na(input1$comment_count)] <- round(mean(input1$comment_count, na.rm = TRUE))
sapply(input1,function(x) sum(is.na(x)))

## replacing missing values with mean 
input1$subscriber[is.na(input1$subscriber)] <- round(mean(input1$subscriber, na.rm = TRUE))
sapply(input1,function(x) sum(is.na(x)))

## replacing missing values with mean 
input1$comment_count[is.na(input1$comment_count)] <- round(mean(input1$comment_count, na.rm = TRUE))
sapply(input1,function(x) sum(is.na(x)))

## replacing missing values with mean 
input1$Trend_day_count[is.na(input1$Trend_day_count)] <- round(mean(input1$Trend_day_count, na.rm = TRUE))
sapply(input1,function(x) sum(is.na(x)))

##checking outliers and treating them
#1
boxplot(input1$subscriber)
summary(input1$subscriber)

#creating upper limit value
upper<-3181914+06+1.5*IQR(input1$subscriber)
upper

#creating lower limit value
lower<-252756+05-1.5*IQR(input1$subscriber)
lower

# upper limit replacement
input1$subscriber[input1$subscriber > upper]<-upper
summary(input1$subscriber)
boxplot(input1$subscriber)

# lower limit replacement
input1$subscribe[input1$subscriber < lower]<-lower
summary(input1$subscriber)

#2
boxplot(input1$Trend_day_count)
summary(input1$Trend_day_count)

#3
boxplot(input1$Tag_count)

#4
boxplot(input1$comment_count)

#5
boxplot(input1$Trend_tag_count)

#6
boxplot(input1$likes)

#7
boxplot(input1$dislike)

model3 <- predict(model2, input1)
input1$Y_p
df<-data.frame(model3)

write.csv(df,file = ("C:/Users/hp/Desktop/hackathon/1234.csv"))

