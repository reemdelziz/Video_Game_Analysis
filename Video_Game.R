# FINAL
install.packages("class")
install.packages("tidyr")
install.packages("caret")
install.packages("e1071")
library(class)
library(tidyr)
library(caret)
library(e1071)

# Load in data using RStudio's built in Import Dataset functionality

####################
## PRE-PROCESSING ##
####################

# Check for any NAs
sum(apply(Game_Sales, 1, anyNA))


# Drop all rows with NA
Game_Sales <- na.omit(Game_Sales)


# Create new column to represent global sales as categorical with 3 levels
Game_Sales$GS_Category <- cut(Game_Sales$Global_Sales, 
                             breaks = c(-Inf, quantile(Game_Sales$Global_Sales, c(1/3, 2/3)), Inf),
                             labels = c("low", "medium", "high"))


# Create new dataframe to preserve original data
Game_Sales2 <- Game_Sales
# Drop extraneous columns
Game_Sales2 <- subset(Game_Sales, select = -c(1, 6, 7, 8, 9, 10))


# Change categorical variables to numerical (KNN only works with numerical predictors)
str(Game_Sales2) #Look at data types of each column in the df
Game_Sales2$Platform <-as.numeric(as.factor(Game_Sales2$Platform))
Game_Sales2$Year_of_Release <-as.numeric(as.factor(Game_Sales2$Year_of_Release))
Game_Sales2$Genre <-as.numeric(as.factor(Game_Sales2$Genre))
Game_Sales2$Publisher <-as.numeric(as.factor(Game_Sales2$Publisher))
Game_Sales2$Developer <-as.numeric(as.factor(Game_Sales2$Developer))
Game_Sales2$Rating <-as.numeric(as.factor(Game_Sales2$Rating))
Game_Sales2$GS_Category <- as.factor(Game_Sales2$GS_Category)
str(Game_Sales2) #Ensure all necessary changes were made correctly


# Scale all non categorical predictors
Game_Sales2[-c(1, 2, 3, 4, 9, 10, 11)] <- scale(Game_Sales2[-c(1, 2, 3, 4, 9, 10, 11)])


#############################
#            KNN            #
#############################
# Randomly split data into 80/20 train and test
n <- nrow(Game_Sales2)
set.seed(1)
part <-sample(1:n, 0.8*n)


train <- Game_Sales2[part,-11]
test <- Game_Sales2[-part,-11]
y.train <- Game_Sales2[part, "GS_Category", drop = TRUE]
y.test <- Game_Sales2[-part, "GS_Category", drop = TRUE]


# Confirm all dimensions are as expected
dim(train)
dim(test)
length(y.train)


# Predictions
knn.test.err <- numeric(7)
K.set = c(1, 3, 5, 8, 10, 15, 20)
set.seed(1)

for (i in 1:length(K.set)) {
  set.seed(1)
  knn.pred = knn(train, test, y.train, k = K.set[i])
  knn.test.err[i] <- mean(knn.pred != y.test)
  #Compare
  cat("K: ", K.set[i], "\n")
  cat("error: ", knn.test.err[i], "\n")
}


opt <- K.set[which.min(knn.test.err)]
cat("optimal K: ", opt)
cat("Error: ", min(knn.test.err))
set.seed(1)
knn.pred = knn(train, test, y.train, k = opt)
summary(knn.pred)
print(confusionMatrix(table(knn.pred, y.test)))


###############################
#             SVM             #
###############################
categorical_vars = c("Platform", "Year_of_Release", "Genre", "Publisher", "Developer", "Rating")


# Convert all categorical variables into factors
for (var in categorical_vars) {
  if (var %in% colnames(Game_Sales2)) {
    Game_Sales2[[var]] <- factor(Game_Sales2[[var]])
  } else {
    warning(paste(var, "not in dataset."))
  }
}
str(Game_Sales2)


# Randomly split data 80/20
n <- nrow(Game_Sales2)
set.seed(1)
train <- sample(n, 0.8*n)
x.train = Game_Sales2[train, ]
x.test = Game_Sales2[-train, ]


# Produces output in this format
set.seed(1)
svm.pred = svm(formula = GS_Category ~ ., data = x.train, kernel = "linear", cost = 0.1)
svm.pred


# Training error
trainPred = predict(svm.pred, x.train)
table(x.train$GS_Category, trainPred)

# Testing error
testPred = predict(svm.pred, x.test)
table(x.test$GS_Category, testPred)


# IDEALLY:


# LINEAR
set.seed(1)
tune.out <- tune(svm, GS_Category ~., data=x.train,
                 kernel="linear",
                 ranges = list(cost=c(0.01, 0.05, 0.1, 0.5, 1, 10)))
tune.out

# Training
train.pred <- predict(tune.out$best.model, x.train)
mean(train.pred != x.train$GS_Category)
table(x.train$GS_Category, train.pred)

# Testing
test.pred <- predict(tune.out$best.model, x.test)
mean(test.pred != x.test$GS_Category)
table(x.test$GS_Category, test.pred)

plot(tune.out$best.model, data=x.train, Critic_Score ~ User_Count)


# POLYNOMIAL
set.seed(1)
tune.out.poly <- tune(svm, GS_Category ~., data=x.train,
                      kernel="polynomial",
                      ranges = list(cost=c(0.01, 0.05, 0.1, 0.5, 1, 10), degree = 3))
tune.out.poly

# Training 
train.pred <- predict(tune.out.poly$best.model, x.train)
mean(train.pred != x.train$GS_Category)
table(x.train$GS_Category, train.pred)

# Testing
test.pred <- predict(tune.out.poly$best.model, x.test)
mean(test.pred != x.test$GS_Category)
table(x.test$GS_Category, test.pred)

plot(tune.out.poly$best.model, data=x.train, Critic_Score ~ User_Count)


# RADIAL
set.seed(1)
tune.out.radial <- tune(svm, GS_Category ~., data=x.train,
                        kernel="radial",
                        ranges = list(cost=c(0.01, 0.05, 0.1, 0.5, 1, 10), gamma = 2))
tune.out.radial

# Training
train.pred <- predict(tune.out.radial$best.model, x.train)
mean(train.pred != x.train$GS_Category)
table(x.train$GS_Category, train.pred)

# Testing
test.pred <- predict(tune.out.radial$best.model, x.test)
mean(test.pred != x.test$GS_Category)
table(x.test$GS_Category, test.pred)

plot(tune.out.radial$best.model, data=x.train, Critic_Score ~ User_Count)


###################################################
#             TRAINING THE BEST MODEL             #
###################################################
categorical_vars = c("Platform", "Year_of_Release", "Genre", "Publisher", "Developer", "Rating")


# Convert all categorical variables into factors
for (var in categorical_vars) {
  if (var %in% colnames(Game_Sales2)) {
    Game_Sales2[[var]] <- factor(Game_Sales2[[var]])
  } else {
    warning(paste(var, "not in dataset."))
  }
}
str(Game_Sales2)


# No Splitting
n <- nrow(Game_Sales2)
x.entire = Game_Sales2


# Produces output in this format
set.seed(1)
svm.pred = svm(formula = GS_Category ~ ., data = x.entire, kernel = "linear", cost = 0.1)
svm.pred


# Training error
entirePred = predict(svm.pred, x.entire)
table(x.entire$GS_Category, entirePred)


# LINEAR
set.seed(1)
tune.out <- tune(svm, GS_Category ~., data=x.entire,
                 kernel="linear",
                 ranges = list(cost=c(0.01, 0.05, 0.1, 0.5, 1, 10)))
tune.out

# Training
entire.pred <- predict(tune.out$best.model, x.entire)
mean(entire.pred != x.entire$GS_Category)
table(x.entire$GS_Category, entire.pred)


plot(tune.out$best.model, data=x.entire, Critic_Score ~ User_Count)
