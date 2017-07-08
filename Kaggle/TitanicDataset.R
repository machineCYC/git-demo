train <- read.csv("D:/Kaggle/TitanicDataset/train.csv", stringsAsFactors=FALSE)
test <- read.csv("D:/Kaggle/TitanicDataset/test.csv")
library(ggplot2)

# structure of the dataframe (train)
str(train)

plot_Missing <- function(data_in, title = NULL){
  # set missing value (NA) 0, others 1
  temp_df <- as.data.frame(ifelse(is.na(data_in), 0, 1))
  # the variable with more NA will be in front of the order
  temp_df <- temp_df[,order(colSums(temp_df))]
  # create a data frame from all combinations of x and y
  data_temp <- expand.grid(list(x = 1:nrow(temp_df), y = colnames(temp_df)))
  # extract NA in each pair (ith data, jth variable)
  data_temp$m <- as.vector(as.matrix(temp_df))
  # create a data frame from all combinations of x, y and m (NA's mark)
  data_temp <- data.frame(x = unlist(data_temp$x), y = unlist(data_temp$y), m = unlist(data_temp$m))
  # plot the figure
  ggplot(data_temp) + geom_tile(aes(x=x, y=y, fill=factor(m))) +
    scale_fill_manual(values=c("white", "black"), name="Missing\n(0=Yes, 1=No)") + theme_light() + ylab("") +
    xlab("") + ggtitle(title)
}
plot_Missing(train[,colSums(is.na(train))>0])


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

### processing missing value in Age, Embarked and Fare

# grow a tree on the subset of the data with the age values available, and then replace the missing value
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], 
                method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

# use "S" to replace the NA value in Embarked, "S" has the most people
summary(factor(combi$Embarked))
combi$Embarked[is.na(combi$Embarked)] = "S"
combi$Embarked <- factor(combi$Embarked)

# use median Fare to replace the Na value in Fare
summary(combi$Fare)
combi$Fare[which(is.na(combi$Fare))] <- median(combi$Fare, na.rm=TRUE)

set.seed(415)
train <- combi[1:891,]
test <- combi[892:1309,]

library(party) # Conditional inference trees

# Since conditional inference trees are able to handle factors with more levels 
# than Random Forests, so we use FamilyID insteaed of FamilyID2
partyfit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                      Fare + Embarked + Title + FamilySize + FamilyID,
                    data=train, controls=cforest_unbiased(ntree=2000, mtry=3))
# mtry: number of variables to choose
Prediction <- predict(partyfit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "D:\\Kaggle\\TitanicDataset\\RRFresult.csv", row.names = FALSE)
# the scored is 0.81340