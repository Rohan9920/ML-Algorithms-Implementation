from svm import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# X, y = datasets.make_blobs(
#     n_samples=5, n_features=2, centers=2, cluster_std=1.05, random_state=40
# )

X = np.array([[ 0.56085542, -8.37942864], [-1.1004791 , -7.78436803], [ 5.82747431, -3.98304522], [-3.78288052, -9.38303174], [ 6.38839328, -3.32438985]])
y = np.array([0, 0, 1, 0, 1])


X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.2,random_state=42)

clf = SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(clf.w, clf.b)

# Visualizations

def get_hyperplane(x, w, b, offset):
        return (-w[0] * x - b + offset) / w[1]

fig, ax = plt.subplots(1, 1, figsize=(10,6))

plt.set_cmap('PiYG')
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=100, alpha=0.75)
plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", c=y_test, s=100, alpha=0.75)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = get_hyperplane(x0_1, clf.w, clf.b, 0)
x1_2 = get_hyperplane(x0_2, clf.w, clf.b, 0)

x1_1_m = get_hyperplane(x0_1, clf.w, clf.b, -1)
x1_2_m = get_hyperplane(x0_2, clf.w, clf.b, -1)

x1_1_p = get_hyperplane(x0_1, clf.w, clf.b, 1)
x1_2_p = get_hyperplane(x0_2, clf.w, clf.b, 1)

ax.plot([x0_1, x0_2], [x1_1, x1_2], "-", c='k', lw=1, alpha=0.9)
ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "--", c='grey', lw=1, alpha=0.8)
ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "--", c='grey', lw=1, alpha=0.8)

x1_min = np.amin(X[:, 1])
x1_max = np.amax(X[:, 1])
ax.set_ylim([x1_min - 3, x1_max + 3])

for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

plt.show()
