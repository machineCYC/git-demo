from sklearn import datasets
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
print(boston.data.shape)
#print(boston.data)
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)

# 針對不同gamma參數進行fit model，並選擇gamma參數
param_range = np.logspace(-6, -3, 10)
train_loss, test_loss = validation_curve(SVR(), X_train, y_train, param_name = "gamma",
                                    param_range = param_range, 
                                    cv = 10, scoring = "neg_mean_squared_error")

train_loss_mean = -np.mean(train_loss, axis = 1)
test_loss_mean = -np.mean(test_loss, axis = 1)

plt.plot(param_range, train_loss_mean, "o-", color = "r", label = "Training")
plt.plot(param_range, test_loss_mean, "o-", color = "g", label = "Cross_validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc = "best")
plt.show()

# 針對不同 C 參數進行fit model，並選擇 C 參數
param_range = np.logspace(-1, 2, 10)
train_loss, test_loss = validation_curve(SVR(gamma = 0.0003), X_train, y_train,
                                         param_name = "C", param_range = param_range, 
                                    cv = 10, scoring = "neg_mean_squared_error")

train_loss_mean = -np.mean(train_loss, axis = 1)
test_loss_mean = -np.mean(test_loss, axis = 1)

plt.plot(param_range, train_loss_mean, "o-", color = "r", label = "Training")
plt.plot(param_range, test_loss_mean, "o-", color = "g", label = "Cross_validation")

plt.xlabel("C")
plt.ylabel("Loss")
plt.legend(loc = "best")
plt.show()


clf = SVR(kernel = "rbf", C = 50, gamma = 0.0003)
clf.fit(X_train, y_train)

# predict test data
predict = clf.predict(X_test)

# show it on the plot
plt.scatter(y_test, y_test, label = "true data")
plt.scatter(y_test, predict, label = "SVR")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, predict))

