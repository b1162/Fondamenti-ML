import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Impostazioni per una migliore qualità dell'immagine
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Generare un dataset sintetico
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Funzione per visualizzare i dati
def plot_kmeans_step(X, centroids=None, labels=None, iteration=None, final=False):
    plt.figure(figsize=(10, 8))
    
    # Visualizzare i punti
    if labels is None:
        plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.8)
        title = "Dataset iniziale"
    else:
        colors = ListedColormap(['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=colors, s=50, alpha=0.8)
        
        if iteration is not None:
            title = f"Iterazione {iteration}: Assegnazione dei punti ai cluster"
        else:
            title = "Assegnazione finale dei cluster"
    
    # Visualizzare i centroidi
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', 
                   s=200, alpha=1, marker='X', edgecolors='black', linewidths=2)
        
        if iteration is not None:
            title = f"Iterazione {iteration}: Aggiornamento dei centroidi"
    
    plt.title(title, fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Aggiungere una legenda per i centroidi
    if centroids is not None:
        centroid_patch = mpatches.Patch(color='red', label='Centroidi')
        plt.legend(handles=[centroid_patch], loc='upper right')
    
    plt.tight_layout()
    
    # Salvare l'immagine
    if iteration is not None:
        if labels is None:
            plt.savefig(f'/home/ubuntu/corso_ml_non_supervisionato/images/kmeans_step{iteration}_centroids.png', dpi=300)
        else:
            plt.savefig(f'/home/ubuntu/corso_ml_non_supervisionato/images/kmeans_step{iteration}_assignment.png', dpi=300)
    elif final:
        plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/kmeans_final.png', dpi=300)
    else:
        plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/kmeans_initial.png', dpi=300)
    
    plt.close()

# Implementazione di K-means
def kmeans_visualization(X, k=4, max_iterations=4):
    # Passo 1: Visualizzare il dataset iniziale
    plot_kmeans_step(X)
    
    # Passo 2: Inizializzare i centroidi casualmente
    indices = np.random.choice(len(X), k, replace=False)
    centroids = X[indices]
    plot_kmeans_step(X, centroids, iteration=0)
    
    for iteration in range(1, max_iterations + 1):
        # Passo 3: Assegnare ogni punto al centroide più vicino
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        plot_kmeans_step(X, centroids, labels, iteration)
        
        # Passo 4: Aggiornare i centroidi
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Verificare la convergenza
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        plot_kmeans_step(X, centroids, iteration=iteration)
    
    # Visualizzazione finale
    plot_kmeans_step(X, centroids, labels, final=True)
    
    return centroids, labels

# Eseguire K-means con visualizzazione
centroids, labels = kmeans_visualization(X, k=4)

print("Visualizzazione K-means completata con successo!")

# Creare un'immagine che mostra il metodo del gomito
def plot_elbow_method():
    plt.figure(figsize=(10, 6))
    
    # Calcolare SSE per diversi valori di k
    sse = []
    k_range = range(1, 11)
    
    for k in k_range:
        # Inizializzare i centroidi casualmente
        indices = np.random.choice(len(X), k, replace=False)
        centroids = X[indices]
        
        # Eseguire K-means
        for _ in range(10):  # Numero di iterazioni
            # Assegnare ogni punto al centroide più vicino
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Aggiornare i centroidi
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                                     else centroids[i] for i in range(k)])
            
            # Verificare la convergenza
            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids
        
        # Calcolare SSE
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        min_distances = np.min(distances, axis=0)
        sse.append(np.sum(min_distances**2))
    
    # Visualizzare il grafico del gomito
    plt.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Numero di cluster (k)', fontsize=14)
    plt.ylabel('Somma degli errori quadratici (SSE)', fontsize=14)
    plt.title('Metodo del Gomito per determinare k ottimale', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    
    # Evidenziare il "gomito"
    optimal_k = 4  # In questo caso, sappiamo che il valore ottimale è 4
    plt.plot(optimal_k, sse[optimal_k-1], 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
    plt.annotate(f'Gomito (k={optimal_k})', 
                xy=(optimal_k, sse[optimal_k-1]), 
                xytext=(optimal_k+1, sse[optimal_k-1]+5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/kmeans_elbow_method.png', dpi=300)
    plt.close()

# Visualizzare il metodo del gomito
plot_elbow_method()

print("Visualizzazione del metodo del gomito completata con successo!")
