import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Impostazioni per una migliore qualità dell'immagine
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Generare dataset sintetici
def generate_datasets():
    # Dataset 1: Due lune (forma non sferica)
    X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    
    # Dataset 2: Cluster con densità diverse
    X1, _ = make_blobs(n_samples=100, centers=[[0, 0]], cluster_std=0.3, random_state=42)
    X2, _ = make_blobs(n_samples=300, centers=[[2.5, 2.5]], cluster_std=0.7, random_state=42)
    X_density = np.vstack([X1, X2])
    
    # Dataset 3: Cluster con rumore
    X_blobs, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=42)
    # Aggiungere rumore
    noise = np.random.uniform(-3, 3, (50, 2))
    X_noise = np.vstack([X_blobs, noise])
    
    return X_moons, X_density, X_noise

# Funzione per visualizzare i punti e i loro tipi (core, border, noise)
def plot_dbscan_step(X, eps, min_samples, step_name, dataset_name):
    plt.figure(figsize=(12, 10))
    
    # Applicare DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)
    
    # Identificare i punti core
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
    # Identificare i punti border (non core ma appartengono a un cluster)
    border_samples_mask = (clusters != -1) & ~core_samples_mask
    
    # Identificare i punti noise
    noise_samples_mask = clusters == -1
    
    # Visualizzare i punti in base al loro tipo
    if step_name == "initial":
        # Visualizzare solo i punti iniziali
        plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.8)
        title = f"Dataset {dataset_name}: Punti iniziali"
    elif step_name == "eps_neighborhood":
        # Visualizzare i punti con il loro ε-vicinato
        plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.8)
        
        # Visualizzare il ε-vicinato per alcuni punti di esempio
        sample_indices = np.random.choice(len(X), 5, replace=False)
        for idx in sample_indices:
            circle = plt.Circle((X[idx, 0], X[idx, 1]), eps, fill=False, 
                               color='red', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
        
        title = f"Dataset {dataset_name}: ε-vicinato (ε={eps})"
    elif step_name == "core_points":
        # Visualizzare i punti core
        plt.scatter(X[~core_samples_mask, 0], X[~core_samples_mask, 1], 
                   s=50, alpha=0.3, c='gray', label='Non-core')
        plt.scatter(X[core_samples_mask, 0], X[core_samples_mask, 1], 
                   s=80, alpha=0.8, c='red', label='Core points')
        
        title = f"Dataset {dataset_name}: Identificazione dei punti core (MinPts={min_samples})"
    elif step_name == "clusters":
        # Visualizzare i cluster finali
        colors = ListedColormap(['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99FF'])
        
        # Visualizzare i punti noise
        plt.scatter(X[noise_samples_mask, 0], X[noise_samples_mask, 1], 
                   s=50, alpha=0.8, c='black', label='Noise')
        
        # Visualizzare i punti core e border con colori diversi per ogni cluster
        for cluster_id in np.unique(clusters[clusters != -1]):
            cluster_mask = clusters == cluster_id
            
            # Punti core del cluster
            core_cluster_mask = cluster_mask & core_samples_mask
            plt.scatter(X[core_cluster_mask, 0], X[core_cluster_mask, 1], 
                       s=80, alpha=0.8, c=[colors(cluster_id % 5)], 
                       edgecolors='black', linewidths=1)
            
            # Punti border del cluster
            border_cluster_mask = cluster_mask & border_samples_mask
            plt.scatter(X[border_cluster_mask, 0], X[border_cluster_mask, 1], 
                       s=50, alpha=0.5, c=[colors(cluster_id % 5)], 
                       edgecolors='black', linewidths=1)
        
        title = f"Dataset {dataset_name}: Cluster finali (ε={eps}, MinPts={min_samples})"
    
    plt.title(title, fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if step_name != "initial" and step_name != "eps_neighborhood":
        plt.legend()
    
    plt.tight_layout()
    
    # Salvare l'immagine
    plt.savefig(f'/home/ubuntu/corso_ml_non_supervisionato/images/dbscan_{dataset_name}_{step_name}.png', dpi=300)
    plt.close()

# Funzione per visualizzare l'effetto dei parametri di DBSCAN
def plot_dbscan_parameters(X, dataset_name):
    eps_values = [0.2, 0.5, 1.0]
    min_samples_values = [3, 5, 10]
    
    fig, axes = plt.subplots(len(eps_values), len(min_samples_values), figsize=(15, 12))
    
    for i, eps in enumerate(eps_values):
        for j, min_samples in enumerate(min_samples_values):
            # Applicare DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X)
            
            # Contare il numero di cluster (escludendo il rumore)
            n_clusters = len(np.unique(clusters[clusters != -1]))
            
            # Visualizzare i risultati
            colors = ListedColormap(['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99FF'])
            
            # Punti noise
            noise_mask = clusters == -1
            axes[i, j].scatter(X[noise_mask, 0], X[noise_mask, 1], 
                              s=30, alpha=0.5, c='black')
            
            # Punti nei cluster
            for cluster_id in np.unique(clusters[clusters != -1]):
                cluster_mask = clusters == cluster_id
                axes[i, j].scatter(X[cluster_mask, 0], X[cluster_mask, 1], 
                                  s=30, alpha=0.8, c=[colors(cluster_id % 5)])
            
            axes[i, j].set_title(f"ε={eps}, MinPts={min_samples}\nCluster: {n_clusters}")
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.suptitle(f"Effetto dei parametri di DBSCAN - Dataset {dataset_name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Salvare l'immagine
    plt.savefig(f'/home/ubuntu/corso_ml_non_supervisionato/images/dbscan_{dataset_name}_parameters.png', dpi=300)
    plt.close()

# Generare i dataset
X_moons, X_density, X_noise = generate_datasets()

# Visualizzare DBSCAN passo-passo per il dataset delle lune
eps_moons = 0.3
min_samples_moons = 5
plot_dbscan_step(X_moons, eps_moons, min_samples_moons, "initial", "moons")
plot_dbscan_step(X_moons, eps_moons, min_samples_moons, "eps_neighborhood", "moons")
plot_dbscan_step(X_moons, eps_moons, min_samples_moons, "core_points", "moons")
plot_dbscan_step(X_moons, eps_moons, min_samples_moons, "clusters", "moons")

# Visualizzare DBSCAN passo-passo per il dataset con densità diverse
eps_density = 0.5
min_samples_density = 5
plot_dbscan_step(X_density, eps_density, min_samples_density, "initial", "density")
plot_dbscan_step(X_density, eps_density, min_samples_density, "eps_neighborhood", "density")
plot_dbscan_step(X_density, eps_density, min_samples_density, "core_points", "density")
plot_dbscan_step(X_density, eps_density, min_samples_density, "clusters", "density")

# Visualizzare DBSCAN passo-passo per il dataset con rumore
eps_noise = 0.7
min_samples_noise = 5
plot_dbscan_step(X_noise, eps_noise, min_samples_noise, "initial", "noise")
plot_dbscan_step(X_noise, eps_noise, min_samples_noise, "eps_neighborhood", "noise")
plot_dbscan_step(X_noise, eps_noise, min_samples_noise, "core_points", "noise")
plot_dbscan_step(X_noise, eps_noise, min_samples_noise, "clusters", "noise")

# Visualizzare l'effetto dei parametri di DBSCAN
plot_dbscan_parameters(X_moons, "moons")
plot_dbscan_parameters(X_density, "density")
plot_dbscan_parameters(X_noise, "noise")

print("Visualizzazione DBSCAN completata con successo!")
