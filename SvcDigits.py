import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

digits = load_digits()
print(digits.images.shape) # (1797, 8, 8)

X = digits.data   # (1797, 64)
y = digits.target # (1797, )

# 將資料分成2/3訓練資料，1/3測試資料
n_samples = len(y)
X_train = X[:int(2*n_samples/3)] # (1198, 64)
y_train = y[:int(2*n_samples/3)] # (1198, )
X_test = X[int(2*n_samples/3):]  # (599, 64)
y_test = y[int(2*n_samples/3):]  # (599, )

# 針對不同參數進行fit model，選出最恰當的參數值
param_range = np.logspace(-6, -2.3, 5)
train_loss, test_loss = validation_curve(SVC(), X_train, y_train,
                                    param_name = "gamma", param_range = param_range, 
                                    cv = 5, scoring = "neg_mean_squared_error")
                         
train_loss_mean = -np.mean(train_loss, axis = 1)
test_loss_mean = -np.mean(test_loss, axis = 1)

plt.plot(param_range, train_loss_mean, "o-", color = "r", label = "Training")
plt.plot(param_range, test_loss_mean, "o-", color = "g", label = "Cross_validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc = "best")
plt.show()

# fit model and predict data
clf = SVC(gamma = 0.0006)
clf.fit(X_train, y_train) 
predict = clf.predict(X_test)

# 利用Confusion matrix 來觀看預測準確度
print("Confusion matrix:\n%s"  % confusion_matrix(y_test, predict))

# 將Confusion matrix視覺化
plt.imshow(confusion_matrix(y_test, predict), interpolation = "nearest",
           cmap = "bone")
plt.title("Confusion matrix")
plt.colorbar()
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.ylabel("True label")
plt.xlabel("Predict label")
plt.show()

# 計算預測精準度
print(clf.score(X_test, y_test))
print(classification_report(y_test, predict))

# Predict the correct rate is 0.96
