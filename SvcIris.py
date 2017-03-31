import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

# data
iris = load_iris()
print(iris["target_names"])
print(iris["feature_names"]) # sepel萼片 長度和寬度, petal花瓣 長度和寬度
X = iris.data[:, :2] #(150, 2) 只單純看sepel，方便畫圖
y = iris.target #(150, )

# 將資料順序打亂
n_samples = len(y)
np.random.seed(0)
order = np.random.permutation(n_samples)
X = X[order]
y = y[order]

# 將資料分成train part和 test part
X_train = X[:int(n_samples/2)] # (75, 2)
y_train = y[:int(n_samples/2)] # (75, )
X_test = X[int(n_samples/2):]  # (75, 2)
y_test = y[int(n_samples/2):]  # (75, )

xx = np.linspace(X.min(), X.max(), 100)
yy = np.linspace(y.min(), y.max(), 100)
xx, yy = np.meshgrid(xx, yy)
xy = np.c_[xx.ravel(), yy.ravel()]
  # xx.ravel() 將xx每一列以行的方式串成一行
  # np.c_[a, b] 將 a,b 以位置相同的值配成一對，形成座標

# parameters
C = 1
gamma = 0.08

# Support vector classification
classifiers = {"Linear SVC": SVC(kernel = "linear", C = C,
                                 probability = True, random_state = 0),
               "Rbf SVC": SVC(kernel = "rbf",   C = C, gamma = gamma,
                              probability = True, random_state = 0),
               "Poly SVC": SVC(kernel = "poly", C = C, gamma = gamma,
                               probability = True, random_state = 0)}

n_classifiers = len(classifiers)
n_clssses = np.unique(y_test).size
fig = plt.figure(figsize = (9, 9))
for i, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print("accuracy for %s: %f" %(name, score)) # %f 指的是浮點數

    #　計算整個網格上每一點屬於每一類的機率
    probas = clf.predict_proba(xy) # (100*100, 3)

    for k in range(n_clssses):
        
        plt.subplot(n_classifiers, n_clssses, i*n_clssses + k + 1)
        plt.title("Class %d" %k) # %d 指的是整數
        if k == 0:
            plt.ylabel(name)
        imshow = plt.imshow(probas[:, k].reshape((100, 100)),
                            extent = (3, 9, 1, 5), origin = "lower")
                            # 3,9 指的是x軸範圍 1,5 指的是y軸範圍  

        #plt.xticks(())
        #plt.yticks(())

        idx = (predict == k)
        plt.scatter(X_test[idx, 0], X_test[idx, 1], marker = "o", c = "red")

ax = plt.axes([0.125, 0.03, 0.75, 0.05])
# (距離左邊多遠, 距離下面多遠, 長度, 寬度)
plt.colorbar(imshow, cax = ax, orientation = "horizontal")

plt.show()



