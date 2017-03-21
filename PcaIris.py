import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# data
iris = load_iris()

print(iris["feature_names"]) # sepel萼片 長度和寬度, petal花瓣 長度和寬度
X = iris.data #(150, 4)
y = iris.target #(150, )

# 2D plot
# Sepal length vs Sepal width
plt.figure(1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c = "blue")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c = "red")
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], c = "green")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

# Petal length vs Petal width
plt.figure(2)
plt.scatter(X[y == 0][:, 2], X[y == 0][:, 3], c = "blue")
plt.scatter(X[y == 1][:, 2], X[y == 1][:, 3], c = "red")
plt.scatter(X[y == 2][:, 2], X[y == 2][:, 3], c = "green")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

# features reduction with PCA
pca = PCA(n_components = 3)
X_reduced = pca.fit_transform(X)

# 3D plot
fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c = y,
           cmap = plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")
plt.show()
