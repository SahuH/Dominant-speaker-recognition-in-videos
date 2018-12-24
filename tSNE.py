#tSNE

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

X_test = np.load('./7class/numpy_test/X_test.npy')
y_test = np.load('./7class/numpy_test/y_test.npy')
print('Test data loaded')
X_test = X_test[:100]
y_test = y_test[:100]
pca = PCA(n_components=50)
pca.fit(X_test)
X_test1 = pca.transform(X_test)
print(X_test1.shape)
print('PCA Done')

tsne = TSNE(n_components=2)
X_test2 = tsne.fit_transform(X_test1)
print(X_test2.shape)
print('tsne Done')

plt.scatter(X_test2[:,0], X_test2[:,1], marker='o', c=y_test)
plt.title('First Test Data after tSNE')
plt.savefig('First Test Data plot.png')