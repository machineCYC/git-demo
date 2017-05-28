### Titanic dataset analysis (Kaggle)
### ---
### Exploratory data analysis (EDA)
### ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# import data
data = pd.read_csv("train.csv")

### exploratory data analysis (EDA) ###
#print(data.head())
#print(data.describe())
data["Age"].fillna(data["Age"].median(), inplace=True)# fill in missing values
#print(data.describe())

# survival based on the gender
survived_sex = data[data["Survived"] == 1]["Sex"].value_counts()
dead_sex = data[data["Survived"] == 0]["Sex"].value_counts()
df = pd.DataFrame([survived_sex, dead_sex])
df.index = ["Survived", "Dead"]
ax = df.plot(kind="bar", stacked=True, figsize=(10, 6), width=0.3)
ax.set_ylabel("Number of passengers")
ax.legend(("female", "male"), loc="upper left")

# survival with the age
figure = plt.figure(figsize=(10, 6))
plt.hist([data[data["Survived"] == 1]["Age"], data[data["Survived"] == 0]["Age"]], stacked=True, color=["g", "r"],
         bins=30, label=["Survived", "Dead"])
plt.xlabel("Age")
plt.ylabel("Number of passengers")
plt.legend()

# survival with the fare ticket
figure = plt.figure(figsize=(10, 6))
plt.hist([data[data["Survived"] == 1]["Fare"], data[data["Survived"] == 0]["Fare"]], stacked=True, color=["g", "r"], bins=30, label=["Survived", "Dead"])
plt.xlabel("Fare")
plt.ylabel("Number of passengers")
plt.legend()

# survival based on the embarkation
survived_embark = data[data["Survived"] == 1]["Embarked"].value_counts()
dead_embark = data[data["Survived"] == 0]["Embarked"].value_counts()
df = pd.DataFrame([survived_embark, dead_embark])
df.index = ["Survived", "Dead"]
ax = df.plot(kind="bar", stacked=True, figsize=(10, 6), width=0.3)
ax.set_ylabel("Number of passengers")
ax.legend(("Embarked S", "Embarked C", "Embarked Q"), loc="best")

# combine the age, the fare and the survival 
plt.figure(figsize=(10, 6))
ax = plt.subplot()
ax.scatter(data[data["Survived"] == 1]["Age"], data[data["Survived"] == 1]["Fare"], c="green", s=20)
ax.scatter(data[data["Survived"] == 0]["Age"], data[data["Survived"] == 0]["Fare"], c="red", s=20)
ax.set_xlabel("Age")
ax.set_ylabel("Fare")
ax.legend(("survived", "dead"), scatterpoints=1, loc="upper right", fontsize=15)

# ticket fare with the passenger class
plt.figure(figsize=(10, 6))
ax = plt.subplot()
ax.set_ylabel("Average fare")
data.groupby("Pclass").mean()["Fare"].plot(kind="bar", color="orange", ax=ax)

# age density plot with the passenger class
plt.figure(figsize=(10, 6))
data.Age[data.Pclass == 1].plot(kind="kde")
data.Age[data.Pclass == 2].plot(kind="kde")
data.Age[data.Pclass == 3].plot(kind="kde")
plt.xlabel("Age")
plt.legend(("1class", "2class", "3class"), loc="best")

# survival with the passenger class
survived_pclass = data.Pclass[data.Survived == 1].value_counts()
dead_pclass = data.Pclass[data.Survived == 0].value_counts()
df = pd.DataFrame({"Survived": survived_pclass, "dead": dead_pclass})
df.plot(kind="bar", stacked=True, color=["green", "red"], figsize=(10, 6))
plt.xlabel("Passenger class")
plt.ylabel("Number of passengers")

plt.show()

# define a print function that asserts whether or not a feature has been processed
def status(feature):
    print("Processing", feature, ": ok")

def get_combined_data():
    # reading train data
    train = pd.read_csv("train.csv")

    # reading test data
    test = pd.read_csv("test.csv")

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop("Survived", 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop("index", inplace=True, axis=1)

    return combined

combined = get_combined_data()

combined.info()

# Extracting the passenger titles
def get_titles():
    global combined

    # we extract the title from each name
    combined["Title"] = combined["Name"].map(lambda name: name.split(",")[1].split(".")[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"

    }

    # we map each title
    combined["Title"] = combined.Title.map(Title_Dictionary)

get_titles()
# print(combined.head())

# Processing the ages
grouped = combined.groupby(["Sex", "Pclass", "Title"])

def process_age():
    global combined

    # a function that fills the missing values of the Age variable
    def fillAges(row):
        if row["Sex"] == "female" and row["Pclass"] == 1:
            if row["Title"] == "Miss":
                return 30
            elif row["Title"] == "Mrs":
                return 45
            elif row["Title"] == "Officer":
                return 49
            elif row["Title"] == "Royalty":
                return 39

        elif row["Sex"] == "female" and row["Pclass"] == 2:
            if row["Title"] == "Miss":
                return 20
            elif row["Title"] == "Mrs":
                return 30

        elif row["Sex"] == "female" and row["Pclass"] == 3:
            if row["Title"] == "Miss":
                return 18
            elif row["Title"] == "Mrs":
                return 31

        elif row["Sex"] == "male" and row["Pclass"] == 1:
            if row["Title"] == "Master":
                return 6
            elif row["Title"] == "Mr":
                return 41.5
            elif row["Title"] == "Officer":
                return 52
            elif row["Title"] == "Royalty":
                return 40

        elif row["Sex"] == "male" and row["Pclass"] == 2:
            if row["Title"] == "Master":
                return 2
            elif row["Title"] == "Mr":
                return 30
            elif row["Title"] == "Officer":
                return 41.5

        elif row["Sex"] == "male" and row["Pclass"] == 3:
            if row["Title"] == "Master":
                return 6
            elif row["Title"] == "Mr":
                return 26

    combined.Age = combined.apply(lambda r: fillAges(r) if np.isnan(r["Age"]) else r["Age"], axis=1)

    status("age")

process_age()
#print(combined.info())

# Processing the names
def process_names():
    global combined
    # we clean the Name variable
    combined.drop("Name", axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined["Title"], prefix="Title")
    combined = pd.concat([combined, titles_dummies], axis=1)

    # removing the title variable
    combined.drop("Title", axis=1, inplace=True)

    status("names")

