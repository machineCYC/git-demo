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


