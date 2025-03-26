---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Algoritmi di Clustering
## Corso di Machine Learning: Apprendimento Non Supervisionato

---

# Indice

1. Introduzione al Clustering
2. K-means Clustering
3. Clustering Gerarchico
4. DBSCAN
5. Gaussian Mixture Models
6. Valutazione dei Risultati di Clustering
7. Applicazioni Pratiche

---

# 1. Introduzione al Clustering

- **Definizione**: Il clustering è una tecnica di apprendimento non supervisionato che raggruppa oggetti simili in cluster
- **Obiettivo**: Trovare strutture intrinseche nei dati senza etichette predefinite
- **Principio di base**: Massimizzare la similarità all'interno dei cluster e minimizzare la similarità tra cluster diversi

---

# Tipi di Algoritmi di Clustering

- **Clustering partizionale**: Divide i dati in gruppi non sovrapposti (es. K-means)
- **Clustering gerarchico**: Crea una decomposizione gerarchica dei dati (es. Agglomerativo)
- **Clustering basato sulla densità**: Identifica regioni dense di punti (es. DBSCAN)
- **Clustering basato su modelli**: Assume che i dati siano generati da una distribuzione (es. GMM)
- **Clustering basato su griglia**: Divide lo spazio in celle (es. STING)

---

# Misure di Distanza nel Clustering

- **Distanza euclidea**: 
  ![Distanza euclidea](images/formula_euclidean.png)

- **Distanza di Manhattan**: 
  ![Distanza di Manhattan](images/formula_manhattan.png)

- **Distanza di Minkowski**: 
  ![Distanza di Minkowski](images/formula_minkowski.png)

- **Distanza del coseno**: 
  ![Distanza del coseno](images/formula_cosine.png)

---

# 2. K-means Clustering

- **Idea di base**: Partizionare i dati in K cluster, dove ogni punto appartiene al cluster con il centroide più vicino
- **Algoritmo**:
  1. Inizializzare K centroidi (casualmente o con metodi come K-means++)
  2. Assegnare ogni punto al centroide più vicino
  3. Ricalcolare i centroidi come media dei punti assegnati
  4. Ripetere i passi 2-3 fino alla convergenza

---

# K-means: Esempio Passo-Passo

![height:500px](images/kmeans_initial.png)

---

# K-means: Inizializzazione dei Centroidi

![height:500px](images/kmeans_step0_centroids.png)

---

# K-means: Prima Iterazione - Assegnazione

![height:500px](images/kmeans_step1_assignment.png)

---

# K-means: Prima Iterazione - Aggiornamento Centroidi

![height:500px](images/kmeans_step1_centroids.png)

---

# K-means: Seconda Iterazione - Assegnazione

![height:500px](images/kmeans_step2_assignment.png)

---

# K-means: Seconda Iterazione - Aggiornamento Centroidi

![height:500px](images/kmeans_step2_centroids.png)

---

# K-means: Risultato Finale

![height:500px](images/kmeans_final.png)

---

# K-means: Vantaggi e Svantaggi

**Vantaggi**:
- Semplice da implementare e comprendere
- Efficiente su grandi dataset
- Garantisce la convergenza

**Svantaggi**:
- Richiede di specificare K a priori
- Sensibile all'inizializzazione dei centroidi
- Assume cluster di forma sferica e dimensioni simili
- Sensibile agli outlier

---

# K-means++: Miglioramento dell'Inizializzazione

- **Problema**: L'inizializzazione casuale può portare a risultati subottimali
- **Soluzione K-means++**:
  1. Scegliere il primo centroide casualmente
  2. Per ogni punto, calcolare la distanza dal centroide più vicino
  3. Scegliere il prossimo centroide con probabilità proporzionale al quadrato della distanza
  4. Ripetere i passi 2-3 fino a selezionare K centroidi

---

# Come Scegliere il Numero Ottimale di Cluster K?

- **Metodo del gomito (Elbow Method)**
  - Grafico della somma degli errori quadratici (SSE) vs. K
  - Scegliere K dove la curva forma un "gomito"

![height:350px](images/kmeans_elbow_method.png)

---

# 3. Clustering Gerarchico

- **Approcci**:
  - **Agglomerativo (bottom-up)**: Inizia con ogni punto come cluster e unisce progressivamente
  - **Divisivo (top-down)**: Inizia con un unico cluster e divide progressivamente

- **Algoritmo Agglomerativo**:
  1. Iniziare con ogni punto come cluster separato
  2. Calcolare la matrice di distanza tra tutti i cluster
  3. Unire i due cluster più vicini
  4. Aggiornare la matrice di distanza
  5. Ripetere i passi 3-4 fino a ottenere un singolo cluster

---

# Metodi di Linkage nel Clustering Gerarchico

- **Single linkage**: Distanza minima tra punti di cluster diversi
  ![Single linkage](images/formula_single_linkage.png)

- **Complete linkage**: Distanza massima tra punti di cluster diversi
  ![Complete linkage](images/formula_complete_linkage.png)

- **Average linkage**: Media delle distanze tra tutte le coppie di punti
  ![Average linkage](images/formula_average_linkage.png)

- **Ward's method**: Minimizza l'aumento della varianza intra-cluster

---

# Dendrogramma: Visualizzazione del Clustering Gerarchico

![height:450px](https://upload.wikimedia.org/wikipedia/commons/a/ad/Hierarchical_clustering_simple_diagram.svg)

---

# Vantaggi e Svantaggi del Clustering Gerarchico

**Vantaggi**:
- Non richiede di specificare il numero di cluster a priori
- Produce una rappresentazione gerarchica (dendrogramma)
- Applicabile a qualsiasi tipo di attributo

**Svantaggi**:
- Complessità computazionale elevata: O(n²log n) o O(n³)
- Non può correggere fusioni o divisioni errate
- Sensibile al rumore e agli outlier

---

# 4. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

- **Idea di base**: Identifica cluster come regioni dense di punti separate da regioni a bassa densità
- **Parametri**:
  - ε (eps): Raggio del vicinato
  - MinPts: Numero minimo di punti in un vicinato per formare un core point

- **Tipi di punti**:
  - **Core point**: Ha almeno MinPts punti nel suo ε-vicinato
  - **Border point**: Non è un core point ma è nel vicinato di un core point
  - **Noise point**: Non è né core né border point

---

# DBSCAN: Esempio Passo-Passo

## Dataset iniziale

![height:450px](images/dbscan_moons_initial.png)

---

# DBSCAN: ε-vicinato

![height:450px](images/dbscan_moons_eps_neighborhood.png)

---

# DBSCAN: Identificazione dei Punti Core

![height:450px](images/dbscan_moons_core_points.png)

---

# DBSCAN: Formazione dei Cluster

![height:450px](images/dbscan_moons_clusters.png)

---

# DBSCAN: Effetto dei Parametri

![height:450px](images/dbscan_moons_parameters.png)

---

# Algoritmo DBSCAN

1. Etichettare tutti i punti come non visitati
2. Per ogni punto non visitato p:
   - Marcare p come visitato
   - Trovare tutti i punti nel ε-vicinato di p
   - Se |ε-vicinato| < MinPts, marcare p come rumore
   - Altrimenti, creare un nuovo cluster con p e aggiungere tutti i punti raggiungibili da p

---

# DBSCAN: Gestione di Cluster con Densità Diverse

![height:450px](images/dbscan_density_clusters.png)

---

# DBSCAN: Robustezza agli Outlier

![height:450px](images/dbscan_noise_clusters.png)

---

# Vantaggi e Svantaggi di DBSCAN

**Vantaggi**:
- Non richiede di specificare il numero di cluster a priori
- Può trovare cluster di forme arbitrarie
- Robusto agli outlier
- Efficiente su grandi dataset

**Svantaggi**:
- Difficoltà nella scelta dei parametri ε e MinPts
- Non gestisce bene cluster con densità variabile
- Problemi con dataset ad alta dimensionalità

---

# 5. Gaussian Mixture Models (GMM)

- **Idea di base**: Modellare i dati come una miscela di distribuzioni gaussiane multivariate
- **Assunzione**: Ogni cluster è rappresentato da una distribuzione gaussiana
- **Parametri per ogni componente k**:
  - μₖ: Media
  - Σₖ: Matrice di covarianza
  - πₖ: Peso della componente (probabilità a priori)

---

# Algoritmo EM per GMM

**Expectation-Maximization (EM)**:
1. **Inizializzazione**: Inizializzare i parametri μₖ, Σₖ, πₖ
2. **E-step**: Calcolare la probabilità che ogni punto appartenga a ciascuna componente
3. **M-step**: Aggiornare i parametri in base alle probabilità calcolate
4. Ripetere E-step e M-step fino alla convergenza

---

# Vantaggi e Svantaggi di GMM

**Vantaggi**:
- Fornisce un modello probabilistico completo
- Flessibile nella forma dei cluster (attraverso la matrice di covarianza)
- Assegna probabilità di appartenenza ai cluster

**Svantaggi**:
- Sensibile all'inizializzazione
- Può convergere a massimi locali
- Richiede di specificare il numero di componenti
- Assume che i dati seguano una distribuzione gaussiana

---

# 6. Valutazione dei Risultati di Clustering

## Metriche Interne (senza etichette vere)

- **Indice di Silhouette**: Misura quanto un punto è simile al suo cluster rispetto agli altri
- **Indice di Davies-Bouldin**: Rapporto tra dispersione intra-cluster e separazione inter-cluster
- **Indice di Calinski-Harabasz**: Rapporto tra varianza inter-cluster e intra-cluster
- **Coefficiente di Dunn**: Rapporto tra distanza minima inter-cluster e distanza massima intra-cluster

---

# Valutazione dei Risultati di Clustering (cont.)

## Metriche Esterne (con etichette vere)

- **Rand Index**: Percentuale di decisioni corrette (stessa/diversa classe)
- **Adjusted Rand Index**: Versione corretta del Rand Index
- **Mutual Information**: Quantifica l'informazione condivisa tra clustering e etichette vere
- **Normalized Mutual Information**: Versione normalizzata della Mutual Information
- **V-measure**: Media armonica di completezza e omogeneità

---

# 7. Applicazioni Pratiche del Clustering

- **Segmentazione dei clienti**: Identificare gruppi di clienti con comportamenti simili
- **Analisi di immagini**: Segmentazione di immagini, riconoscimento di oggetti
- **Bioinformatica**: Raggruppamento di geni con espressione simile
- **Sistemi di raccomandazione**: Raggruppare utenti o prodotti simili
- **Analisi dei social network**: Identificare comunità
- **Compressione dei dati**: Quantizzazione vettoriale
- **Rilevamento di anomalie**: Identificare punti che non appartengono a nessun cluster

---

# Riepilogo: Confronto tra Algoritmi di Clustering

| Algoritmo | Forma dei cluster | Scalabilità | Parametri richiesti | Robustezza agli outlier |
|-----------|-------------------|-------------|---------------------|-------------------------|
| K-means | Sferico | Alta | K | Bassa |
| Gerarchico | Arbitraria | Bassa | Soglia di taglio | Media |
| DBSCAN | Arbitraria | Media | ε, MinPts | Alta |
| GMM | Ellittico | Media | K | Bassa |

---

# Domande?

![height:450px](https://miro.medium.com/max/1400/1*tQTcGVoYIX8UZd7q2zCDQw.png)

---

# Riferimenti

- Scikit-learn: Clustering. https://scikit-learn.org/stable/modules/clustering.html
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning.
- Aggarwal, C. C., & Reddy, C. K. (2013). Data clustering: algorithms and applications.
- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise.
