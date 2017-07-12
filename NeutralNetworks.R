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
#
#Weights----------------------------
# By comparing the output to a known label, we adjust the weights (beginning from random initialization values)
# And repeat until the max. number of allowed iterations/ acceptable error rate is reached. 
#
#Creating a neural network-----------
# Add layers of perceptrons together, consisting:
# Input layer and output layer
# Hidden layers that do not directly have feature inputs or outputs 
#
#Creating a neural network in R===========================

# Utilise Boston dataset, which contains housing values in Boston. Goal: predict median value of homes
set.seed(500)
library(MASS)
data <- Boston 
?Boston 
head(data)

# Check for missing datapoints 
apply(data, 2, function(x) sum(is.na(x))) #check for missing datapoints 

# Randomly split data into train and test set 
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

# generate linear model for comparison purposes 
lm.fit <- glm(medv~., data= train) #glm used instead of lm, useful when cross validating linear model
summary(lm.fit)
pr.lm <- predict(lm.fit,test)
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test) #Mean Squared Error as a measure of how far predictions are from real data 

# Preprocessing data 
## Normalize data. Different methods can be used (z-normalization, min-max). 
## We use the min-max method and scale the data in interval [0, 1]
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train.scaled <- scaled[index,]
test.scaled <- scaled[-index,]

# Fitting the neural net
## Parameters: : How many layers and neurons?
## We will use 2 hidden layers with configuration 13:5:3:1 (input: hidden layers: output)



