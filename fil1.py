import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from scipy.cluster.vq import kmeans, vq
import tensorflow as tf
from tensorflow import keras

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the silhouette score to find the optimal number of clusters
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, y_kmeans))

# Plotting the silhouette scores
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Training the K-Means model on the dataset
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Adding 2 as the range starts from 2
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('Clusters of customers (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Neural Network-based Clustering with TensorFlow/Keras
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_scaled, y_kmeans, epochs=50, verbose=0)

# Predict cluster labels using the neural network
y_nn = np.argmax(model.predict(X_scaled), axis=1)

# Visualizing the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_nn, cmap='viridis', s=50)
plt.title('Clusters of customers (Neural Network)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Clustering with PyTorch
X_torch = torch.tensor(X_scaled, dtype=torch.float32)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

# Training the autoencoder
num_epochs = 50
for epoch in range(num_epochs):
    outputs = autoencoder(X_torch)
    loss = criterion(outputs, X_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Obtain cluster labels using k-means on the autoencoder's encoded output
encoded_output = autoencoder.encoder(X_torch).detach().numpy()
kmeans_centroids, kmeans_labels = kmeans(encoded_output, optimal_clusters)
y_autoencoder = vq(encoded_output, kmeans_centroids)[0]

# Visualizing the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_autoencoder, cmap='viridis', s=50)
plt.title('Clusters of customers (Autoencoder)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
