#Neural Networks=======================================
# Neural Networks are a machine learning framework that attempts to mimic the the learning pattern of natural biological neural networks. 
# Biological neural networks have interconnected neurons with dendrites that receive inputs, 
# then based on these inputs they produce an output signal through an axon to another neuron. 
# We will try to mimic this process through the use of Artificial Neural Networks (ANN).
#
#Perceptron------------------------
# The process of creating a neural network begins with the most basic form, a single perceptron.
# A Perceptron has one or more inputs, an activation function, and a single output.
# It receives inputs, multiplies them by some weight, then passes them into an activation function to produce output. 
# Examples of activation functions: logistic function, trigometric function, step function. 
# A bias is added to the perception to avoid issues where all inputs could be equal to zero. 
# Think bias = intercept
#
#Weights----------------------------
# By comparing the output to a known label, we adjust the weights (beginning from random initialization values)
# And repeat until the max. number of allowed iterations/ acceptable error rate is reached. 
# Think weight = slope 
#
#Creating a neural network-----------
# Add layers of perceptrons together, consisting:
# Input layer and output layer
# Hidden layers that do not directly have feature inputs or outputs 
#

#Creating a neural network in R===========================

# Preparation------------------------

# Utilise Boston dataset, which contains housing values in Boston. Goal: predict median value of homes
set.seed(500)
library(MASS)
data <- Boston 
?Boston 
head(data)

# Check for missing datapoints 
apply(data, 2, function(x) sum(is.na(x))) #check for missing datapoints 

# Train-test random splitting for linear model
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

# Fitting linear model for comparison purposes 
lm.fit <- glm(medv~., data= train) #glm used instead of lm, useful when cross validating linear model
summary(lm.fit)

# Predicted data from lm
pr.lm <- predict(lm.fit,test)

# Test MSE (MSE measure of how far predictions are from real data)
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)
pr.lm <- predict(lm.fit,test)

# Neural Net Fitting------------------
## Preprocessing data 

# Normalize data. Different methods can be used (z-normalization, min-max). 
## We use the min-max method and scale the data in interval [0, 1]
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

# Train-test split
train.scaled <- scaled[index,]
test.scaled <- scaled[-index,]

# Fitting the neural net
## Parameters: : How many layers and neurons?
## We will use 2 hidden layers with configuration 13:5:3:1 (input: hidden layers: output)

library(neuralnet)
?neuralnet 
names <- names(train.scaled)
names.all <- as.formula(paste("medv ~", paste(names[!names %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train.scaled,hidden=c(5,3),linear.output=T)
plot(nn)

# Woah! What happened? 
## NeuralNet package in R has trained the data using chaining, backpropagation, gradient descent
## to optimize the weights 

#Now we can try to predict the values for the test set and calculate the MSE. 
?compute
pr.nn <- compute(nn,test.scaled[,1:13]) #using test data against nn

#Remember that the net will output a normalized prediction,
# so we need to scale it back in order to make a meaningful comparison
pr.nn_ <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test.r <- (test.scaled$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test.scaled)

#printing both the linear and nn MSE side by side (lower the better)
print(paste(MSE.lm,MSE.nn))

#plotting both to compare visually
par(mfrow=c(1,2))
plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test$medv,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

--------------------------------------------------------------------------------------------------------
#CROSS VALIDATION
--------------------------------------------------------------------------------------------------------
#A huge part of NN is cross validating your network
#The basic idea is repeating what we've done multiple times.

library(boot)

#repeating lineral model for MSE
set.seed(201)
lm.fit <- glm(medv~.,data=data)
cv.glm(data,lm.fit,K=10)$delta[1]

#using for loop to repear NN model for MSE
set.seed(451)
cv.error <- NULL
k <- 10

#used pbar = fancy way of showing the progress of for loop
library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)
for(i in 1:k){
  index <- sample(1:nrow(data),round(0.9*nrow(data)))
  train.cv <- scaled[index,]
  test.cv <- scaled[-index,]
  
  nn <- neuralnet(f,data=train.cv,hidden=c(5,2),linear.output=T)
  
  pr.nn <- compute(nn,test.cv[,1:13])
  pr.nn <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
  
  test.cv.r <- (test.cv$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
  
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}
boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN', horizontal=TRUE)