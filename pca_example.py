#if not previously done, you need to install numpy, sklearn and matplotlib with the following command line:
#pip install numpy sklearn matplotlib

#import functions and classes to use in this example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#load the Iris dataset, formed by a data-matrix X and a target column-vector y
iris = load_iris() #have a look to all iris attributes if you are curious about the Iris dataset
X = iris.data
y = iris.target
print(f'The data-table is formed by {X.shape[0]} records, each one with {X.shape[1]} features.')
print(f'The records are labeled with {np.unique(y).size} different labels.')

#we standardize the data-table, so after that any column has mean=0 and stdev=1 (note that this step is not always required)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f'The data-table has been standardized. In fact, means are {np.mean(X_scaled,axis=0)} and standard deviations are {np.std(X_scaled,axis=0)}')

#run PCA without reducing the number of features in order to see how many principal components explain most of the variance
pca = PCA()
pca.fit(X_scaled)
X_transformed = pca.transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
print(f'Percentage of explained variance of all the principal components = {explained_variance}')

#plot the explained variance as a bar-plot
plt.figure(figsize=(6,4))
plt.bar(x=range(4),
        height=explained_variance,
        label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#now we saw that 2 features explain more than 95% of the variance, so we reduce the dimensionality of the data-table to 2
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_scaled) #it is equivalent to first calling fit() and then transform()
print(f'The shape of the reduced data-table is {X_transformed.shape}')

#plot the data-table in a 2D plane... we also use the target-vector y to color the points
plt.clf()
scatter = plt.scatter(x=X_transformed[:,0],
                      y=X_transformed[:,1],
                      c=y)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.legend(handles=scatter.legend_elements()[0], 
           labels=iris.target_names.tolist(),
           title='labels',
           loc='best')
plt.tight_layout()
plt.show()

#print the loadings, i.e., the coefficients in the linear combinations which make up any principal component
loadings = pca.components_
print(f'Loadings for PC1 = {loadings[0]}')
print(f'Loadings for PC2 = {loadings[1]}')