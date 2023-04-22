"""
from:
https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

# %matplotlib inline

mnist = fetch_openml("mnist_784")
y1 = mnist.target
y = y1.astype(np.uint8)
n_classes = len(set(y))

Y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Y = label_binarize(mnist.target, classes=[*range(n_classes)])

X_train, X_test, y_train, y_test = train_test_split(mnist.data, Y, random_state=42)
image = np.asarray(X_test)[0]
plt.imshow(image.reshape(28, 28))
plt.show()

clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0))
clf.fit(X_train, y_train)

y_score = clf.predict_proba(X_test)

# precision recall curve
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()
