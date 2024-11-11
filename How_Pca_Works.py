import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Generate random data
num_samples = 100
num_features = 10
Data = np.random.randn(num_samples, num_features)

# Generate random labels (0 or 1) for binary classification
labels = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))


# Plot Original Data 
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(Data[:, 0], Data[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Data (Before PCA)')
plt.colorbar(label='Target Class')
plt.show()


#------------------------------------------------------------------------------------------
# Normalize is essensial in PCA
scaler = StandardScaler()
norml_data = scaler.fit_transform(Data)
#---------------------------------------------------------------------------------------------

#####PCA Manually (4 Steps)

#1 Calculate the covariance matrix of the data
cov_matrix = np.cov(norml_data, rowvar=False)


#2 Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

#3 Sort the eigenvalues and eigenvectors from largest to smallest and rearrangement them 
sort= np.argsort(eigenvalues)[::-1]  # -1 means largest to smallest 
sorted_eigenvalues = eigenvalues[sort]
sorted_eigenvectors = eigenvectors[:, sort]


# 4. Select top eigenvectors
dimention = 2
selected_eigenvectors = sorted_eigenvectors[:, :dimention]       # higher eigenvectors is selected 
    
# Reduce the data dimensions    
Data_reduced = np.dot(norml_data, selected_eigenvectors)
    

# Plot new Data (reconstructed data)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(Data_reduced[:, 0], Data_reduced[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title(' Normal Data (After PCA Manual)')
plt.colorbar(label='Target Class')
plt.show()


#-------------------------------------------------------------------------------------------------------
##### PCA from Library 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(norml_data)


# Plot new Data (reconstructed data) 
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Normal Data (After PCA Library)')
plt.colorbar(label='Target Class')
plt.show()


