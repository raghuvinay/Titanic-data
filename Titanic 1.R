# Set as Working Directory
setwd("~/Desktop/Data Science/Learning/Raw")

# Load data

test = read.csv("test.csv")
train =  read.csv("train.csv")

#Baseline Model = As 61% of the train data pass are dead we assume all the test are dead
test$Survived <- rep(0,nrow(test))
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = 'theyallperish.csv', row.names = FALSE)

#Observed that most of the females survived,So Enhance the model
prop.table(table(train$Sex, train$Survived),1)
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1 
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = 'femalelive.csv', row.names = FALSE)

#Use Age Parameter. Identifying Child
train$Child <- 0
train$Child[train$Age < 18] <- 1
aggregate(Survived ~ Child + Sex, data = train, FUN = function(x) {sum(x)/length(x)})

#Using Fare parameter

train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'
aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

#Observation: Most of Class3 Females missed out on Tickets So Exclude them from the pred

test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = 'femaleliveclass.csv', row.names = FALSE)

#Implementing decision trees

library(rpart)
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train,method = "class")
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RcolorBrewer)
fancyRpartPlot(fit)

#Implement above CART model on Test set

Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = 'myfirstdtree.csv', row.names = FALSE)

#Feature Engineering

test$Survived <- NA
combi <- rbind(train[1:12],test)
# Strings are by default imported as factors, So convert into pure text
combi$Name <- as.character(combi$Name)
# "Braund, Mr. Owen Harris" Extract Mr/Mrs/Master etc..,

strsplit(combi$Name)
combi$Title <- sapply(combi$Name, FUN = function(x) {strsplit(x, split = '[.,]')[[1]][2]})

# Combine unwanted titles
combi$Title[combi$Title %in% c('Mme','Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
combi$Title <- factor(combi$Title)

# Combine family size and create a ID with Surname and Family size

combi$FamilySize <-  combi$SibSp + combi$Parch + 1
combi$Surname <- sapply(combi$Name, FUN = function(x) {strsplit(x, split = '[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep = "")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

# Reconcile train and test data 

train <- combi[1:891,]
test <- combi[892:1309,]
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data=train, method="class")
fancyRpartPlot(fit)
Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = 'newvardtree.csv', row.names = FALSE)

## Random Forest : As random forest cannot deal with missing variables, so replace manually

#Replacing Age
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize
                , data = combi[!is.na(combi$Age),], method = "anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

#Replacing Ensembled
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)

#Replace fare
which(is.na(combi$Fare))
combi$Fare[which(is.na(combi$Fare))] <- median(combi$Fare,na.rm = TRUE)

# Random forest can digest only up to 32 levels, So reduce levels of FamilyID
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

## Using random forest
install.packages('randomForest')
library(randomForest)
# To set the random seed in R
set.seed(415)
train <- combi[1:891,]
test <- combi[892:1309,]
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                      Embarked + Title + FamilySize + FamilyID2, data = train,
                    importance = TRUE, ntree = 2000)
# To Check variable importance
varImpPlot(fit)

Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = 'firstforest.csv', row.names = FALSE)

## Another model: Conditiona Inference Forest
install.packages('party')
library(party)
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                 Embarked + Title + FamilySize + FamilyID,
               data = train, 
               controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = 'conditionforest1.csv', row.names = FALSE)
