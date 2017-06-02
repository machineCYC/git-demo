train <- read.csv("D:/Kaggle/TitanicDataset/train.csv", stringsAsFactors=FALSE)
test <- read.csv("D:/Kaggle/TitanicDataset/test.csv")

# structure of the dataframe (train)
str(train)

### Feature Engineering
test$Survived <- NA
combi <- rbind(train, test)
combi$Name <- as.character(combi$Name)
# split , and . from the name and create the new feature "Title"
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split="[,.]")[[1]][2]})
# delete the space" " in front of the title
combi$Title <- sub(" ", "", combi$Title)

# check the final Title
table(combi$Title)

# combine "Mme" and "Mlle" to "Mlle" 
combi$Title[combi$Title %in% c( "Mme")] <- "Mlle"
# combine "Capt", "Don", "Major" and "Sir" to "Sir" (military titles)
combi$Title[combi$Title %in% c("Capt", "Don", "Major")] <- "Sir"
# combine "Dona", "the Countess", "Jonkheer" and "Lady" to "Lady" (rich folks)
combi$Title[combi$Title %in% c("Dona", "the Countess", "Jonkheer")] <- "Lady"
combi$Title <- factor(combi$Title)

# Check the merged Title
table(combi$Title)

# check structure of the dataframe (combi)
str(combi)

### hypothesis: large families might have trouble sticking together in the panic

# create new feature "FamilySize"
combi$FamilySize <- combi$SibSp + combi$Parch + 1
# extract the passengers¡¦ last names, create new feature "Surname"
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
# create new feature "FamilyID"
combi$FamilyID <- paste(combi$FamilySize, combi$Surname, sep="")
# creat new feature "Small", this mean the family size is two or less 
combi$FamilyID[combi$FamilySize <= 2] <- "Small"

# check FamilyID
table(combi$FamilyID)

# we wanted only family sizes of 3 or more, clean this up
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- "Small"
# Convert to a factor
combi$FamilyID <- factor(combi$FamilyID)

# separate the train set and test set
combi$Sex <- factor(combi$Sex)
train <- combi[1:891,]
test <- combi[892:1309,]

library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
# fit Decision Trees
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, 
             method="class")

# plot Decision Trees
fancyRpartPlot(fit)

Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "D:\\Kaggle\\TitanicDataset\\Rresulttemp.csv", row.names = FALSE)
# the scored is 0.79426