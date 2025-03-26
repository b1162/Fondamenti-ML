---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Tecniche di Riduzione della Dimensionalità
## Corso di Machine Learning: Apprendimento Non Supervisionato

---

# Indice

1. Introduzione alla Riduzione della Dimensionalità
2. Principal Component Analysis (PCA)
3. t-SNE (t-distributed Stochastic Neighbor Embedding)
4. UMAP (Uniform Manifold Approximation and Projection)
5. Autoencoders per la Riduzione della Dimensionalità
6. Selezione vs. Estrazione delle Caratteristiche
7. Applicazioni Pratiche

---

# 1. Introduzione alla Riduzione della Dimensionalità

- **Definizione**: Processo di riduzione del numero di variabili (features) mantenendo le informazioni più rilevanti
- **Motivazioni**:
  - Visualizzazione dei dati
  - Riduzione della complessità computazionale
  - Miglioramento delle prestazioni degli algoritmi
  - Rimozione di caratteristiche ridondanti o irrilevanti
  - Contrasto alla "maledizione della dimensionalità"

---

# La Maledizione della Dimensionalità

- **Problema**: All'aumentare delle dimensioni, i dati diventano sempre più sparsi
- **Conseguenze**:
  - Distanze tra punti tendono a diventare simili
  - Necessità di più dati per generalizzare
  - Aumento della complessità computazionale
  - Overfitting più probabile

![height:300px](https://miro.medium.com/max/1400/1*i0DRwbIOe0LvG-dHSsYWJg.png)

---

# Approcci alla Riduzione della Dimensionalità

- **Metodi lineari**:
  - Principal Component Analysis (PCA)
  - Linear Discriminant Analysis (LDA)
  - Factor Analysis

- **Metodi non lineari**:
  - t-SNE
  - UMAP
  - Isomap
  - Locally Linear Embedding (LLE)
  - Autoencoders

---

# 2. Principal Component Analysis (PCA)

- **Idea di base**: Proiettare i dati su un sottospazio di dimensione inferiore che massimizza la varianza
- **Obiettivo**: Trovare le direzioni (componenti principali) lungo le quali i dati variano maggiormente
- **Assunzioni**: Relazioni lineari tra le variabili

---

# Algoritmo PCA

1. **Standardizzazione**: Normalizzare i dati (media 0, varianza 1)
2. **Calcolo della matrice di covarianza**: $\Sigma = \frac{1}{n-1}X^TX$
3. **Decomposizione in autovalori e autovettori**: $\Sigma = V\Lambda V^T$
4. **Ordinamento degli autovettori**: In base agli autovalori (varianza spiegata)
5. **Selezione delle componenti**: Scegliere i primi k autovettori
6. **Proiezione dei dati**: $X_{ridotto} = XV_k$

---

# Scelta del Numero di Componenti

- **Varianza spiegata**: Percentuale di varianza catturata dalle componenti
  - $\text{Varianza spiegata} = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{n}\lambda_i}$

- **Scree plot**: Grafico degli autovalori in ordine decrescente
  - Cercare il "gomito" nella curva

- **Regola dell'80-90%**: Selezionare componenti fino a spiegare l'80-90% della varianza

---

# Vantaggi e Svantaggi di PCA

**Vantaggi**:
- Semplice da implementare e interpretare
- Efficiente computazionalmente
- Preserva la maggior parte della varianza
- Rimuove la correlazione tra le variabili

**Svantaggi**:
- Assume relazioni lineari
- Sensibile alla scala delle variabili
- Non preserva necessariamente la struttura locale
- Le componenti principali possono essere difficili da interpretare

---

# 3. t-SNE (t-distributed Stochastic Neighbor Embedding)

- **Idea di base**: Preservare la struttura locale dei dati
- **Obiettivo**: Mantenere le similarità tra punti vicini nello spazio originale
- **Caratteristiche**:
  - Metodo non lineare
  - Particolarmente efficace per la visualizzazione
  - Preserva i cluster

---

# Algoritmo t-SNE

1. Calcolare le probabilità condizionali nello spazio originale:
   - $p_{j|i} = \frac{\exp(-||x_i-x_j||^2/2\sigma_i^2)}{\sum_{k \neq i}\exp(-||x_i-x_k||^2/2\sigma_i^2)}$

2. Calcolare le probabilità simmetriche:
   - $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$

3. Calcolare le probabilità nello spazio ridotto usando la distribuzione t di Student:
   - $q_{ij} = \frac{(1+||y_i-y_j||^2)^{-1}}{\sum_{k \neq l}(1+||y_k-y_l||^2)^{-1}}$

4. Minimizzare la divergenza KL tra P e Q:
   - $KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$

---

# Parametri Importanti di t-SNE

- **Perplexity**: Bilancia l'attenzione tra struttura locale e globale
  - Tipicamente tra 5 e 50
  - Influenza la larghezza della distribuzione gaussiana

- **Numero di iterazioni**: Influenza la convergenza
  - Tipicamente 1000-2000 iterazioni

- **Learning rate**: Controlla la velocità di apprendimento
  - Valori tipici: 10-1000

---

# Vantaggi e Svantaggi di t-SNE

**Vantaggi**:
- Eccellente per la visualizzazione dei dati
- Preserva la struttura locale e i cluster
- Gestisce bene relazioni non lineari

**Svantaggi**:
- Computazionalmente costoso: O(n²)
- Non preserva la struttura globale
- Risultati possono variare con diverse inizializzazioni
- Non adatto per ridurre a dimensioni > 3
- Non è una trasformazione generalizzabile a nuovi dati

---

# 4. UMAP (Uniform Manifold Approximation and Projection)

- **Idea di base**: Apprendere la topologia dei dati ad alta dimensione e trovare una rappresentazione a bassa dimensione con topologia simile
- **Fondamenti teorici**: Teoria dei grafi e geometria Riemanniana
- **Obiettivo**: Bilanciare la preservazione della struttura locale e globale

---

# Algoritmo UMAP (semplificato)

1. Costruire un grafo pesato che rappresenta la struttura locale dei dati
2. Ottimizzare un layout a bassa dimensione per preservare la struttura del grafo
3. Minimizzare una funzione di costo che bilancia:
   - Attrazione tra punti vicini
   - Repulsione tra punti lontani

---

# Parametri Importanti di UMAP

- **n_neighbors**: Controlla la struttura locale
  - Valori più alti preservano più struttura globale
  - Valori più bassi preservano più struttura locale

- **min_dist**: Controlla la compattezza della rappresentazione
  - Valori più bassi creano cluster più compatti
  - Valori più alti preservano meglio le distanze globali

- **metric**: Metrica di distanza utilizzata
  - Euclidea, coseno, correlazione, ecc.

---

# Confronto tra UMAP e t-SNE

**UMAP vantaggi rispetto a t-SNE**:
- Più veloce: O(n log n) vs O(n²)
- Migliore preservazione della struttura globale
- Più stabile con diverse inizializzazioni
- Può generalizzare a nuovi dati
- Supporta più metriche di distanza

**Quando usare quale**:
- t-SNE: Quando la visualizzazione e la separazione dei cluster sono prioritarie
- UMAP: Quando velocità e preservazione della struttura globale sono importanti

---

# 5. Autoencoders per la Riduzione della Dimensionalità

- **Idea di base**: Reti neurali che imparano a comprimere i dati e poi a ricostruirli
- **Architettura**:
  - **Encoder**: Comprime i dati in una rappresentazione latente
  - **Bottleneck**: Spazio latente a dimensionalità ridotta
  - **Decoder**: Ricostruisce i dati originali dalla rappresentazione latente

---

# Tipi di Autoencoders

- **Autoencoder semplice**: Fully connected con bottleneck
- **Autoencoder convoluzionale**: Usa layer convoluzionali (per immagini)
- **Autoencoder variazionale (VAE)**: Genera uno spazio latente continuo
- **Autoencoder denoising**: Addestrato a ricostruire dati puliti da input rumorosi
- **Autoencoder sparso**: Impone vincoli di sparsità sullo spazio latente
- **Autoencoder contrattivo**: Penalizza la sensibilità dell'encoder alle variazioni dell'input

---

# Vantaggi e Svantaggi degli Autoencoders

**Vantaggi**:
- Possono catturare relazioni non lineari complesse
- Flessibili e adattabili a diversi tipi di dati
- Possono essere specializzati per compiti specifici
- Generano rappresentazioni utili per altri task

**Svantaggi**:
- Richiedono più dati per l'addestramento
- Computazionalmente più costosi
- Più difficili da interpretare
- Rischio di overfitting

---

# 6. Selezione vs. Estrazione delle Caratteristiche

**Selezione delle caratteristiche**:
- Seleziona un sottoinsieme delle caratteristiche originali
- Mantiene l'interpretabilità
- Esempi: Filter methods, Wrapper methods, Embedded methods

**Estrazione delle caratteristiche**:
- Crea nuove caratteristiche combinando quelle originali
- Può catturare relazioni più complesse
- Esempi: PCA, t-SNE, UMAP, Autoencoders

---

# Metodi di Selezione delle Caratteristiche

- **Filter methods**: Valutano le caratteristiche indipendentemente dal modello
  - Correlazione, Mutual Information, Chi-square test, ANOVA

- **Wrapper methods**: Valutano sottoinsiemi di caratteristiche usando il modello
  - Forward Selection, Backward Elimination, Recursive Feature Elimination

- **Embedded methods**: Selezionano caratteristiche durante l'addestramento del modello
  - LASSO, Ridge Regression, Random Forest importance

---

# 7. Applicazioni Pratiche della Riduzione della Dimensionalità

- **Visualizzazione dei dati**: Rappresentare dati multidimensionali in 2D/3D
- **Preprocessing per ML**: Migliorare le prestazioni di algoritmi sensibili all'alta dimensionalità
- **Compressione dei dati**: Ridurre lo spazio di archiviazione
- **Estrazione di caratteristiche**: Generare rappresentazioni per altri task
- **Rimozione del rumore**: Filtrare componenti irrilevanti
- **Analisi di immagini**: Riconoscimento facciale, compressione
- **Analisi di testi**: Topic modeling, word embeddings
- **Genomica**: Analisi di espressione genica

---

# Quale Tecnica Scegliere?

| Tecnica | Preserva struttura | Complessità | Interpretabilità | Visualizzazione | Generalizzazione |
|---------|-------------------|-------------|-----------------|----------------|-----------------|
| PCA | Globale (lineare) | Bassa | Alta | Buona | Sì |
| t-SNE | Locale | Alta | Bassa | Eccellente | No |
| UMAP | Locale e globale | Media | Bassa | Eccellente | Sì |
| Autoencoders | Dipende | Alta | Bassa | Buona | Sì |

---

# Riepilogo

- La riduzione della dimensionalità è fondamentale per gestire dati complessi
- PCA è semplice ed efficace per relazioni lineari
- t-SNE eccelle nella visualizzazione e preservazione dei cluster
- UMAP offre un buon compromesso tra struttura locale e globale
- Gli autoencoders sono flessibili e potenti per relazioni complesse
- La scelta della tecnica dipende dall'obiettivo e dalla natura dei dati

---

# Domande?

![height:450px](https://miro.medium.com/max/1400/1*QS7lV3q12CQPj0V9lKHbGQ.png)

---

# Riferimenti

- Scikit-learn: Dimensionality Reduction. https://scikit-learn.org/stable/modules/decomposition.html
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
