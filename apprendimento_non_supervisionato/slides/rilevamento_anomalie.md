---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Rilevamento di Anomalie
## Corso di Machine Learning: Apprendimento Non Supervisionato

---

# Indice

1. Introduzione al Rilevamento di Anomalie
2. Approcci Statistici
3. Metodi Basati sulla Densità
4. Isolation Forest
5. One-Class SVM
6. Rilevamento di Anomalie Basato su Clustering
7. Rilevamento di Anomalie in Serie Temporali
8. Applicazioni Pratiche

---

# 1. Introduzione al Rilevamento di Anomalie

- **Definizione**: Identificazione di osservazioni che deviano significativamente dal comportamento normale
- **Terminologia**: Anomalie, outlier, novità, eccezioni, aberrazioni
- **Tipi di anomalie**:
  - **Puntuali**: Singole istanze anomale
  - **Contestuali**: Anomale in un contesto specifico
  - **Collettive**: Gruppi di istanze anomale

---

# Sfide nel Rilevamento di Anomalie

- **Definizione di "normale"**: Difficoltà nel definire il comportamento normale
- **Confine sfumato**: Distinzione non netta tra normale e anomalo
- **Evoluzione della normalità**: Il comportamento normale può cambiare nel tempo
- **Rumore**: Difficoltà nel distinguere tra anomalie e rumore
- **Etichette limitate**: Spesso mancano esempi di anomalie
- **Dimensionalità**: La maledizione della dimensionalità complica il rilevamento

---

# Approcci al Rilevamento di Anomalie

- **Supervisionato**: Richiede esempi etichettati di normali e anomali
- **Semi-supervisionato**: Richiede solo esempi di dati normali
- **Non supervisionato**: Non richiede etichette, assume che le anomalie siano rare

![height:300px](https://miro.medium.com/max/1400/1*-p-3NWI5aLsVKkQNHGgZJg.png)

---

# 2. Approcci Statistici

- **Idea di base**: Modellare la distribuzione dei dati normali e identificare come anomalie i punti con bassa probabilità
- **Assunzione**: I dati normali seguono una distribuzione statistica nota

---

# Test Statistici per Outlier

- **Z-score**: $z = \frac{x - \mu}{\sigma}$
  - Anomalie: $|z| > 3$ (assumendo distribuzione normale)

- **Test di Grubbs**: Per identificare un singolo outlier
  - $G = \frac{\max_i |x_i - \bar{x}|}{s}$

- **Test di Dixon**: Per campioni piccoli
  - $Q = \frac{|x_n - x_{n-1}|}{|x_n - x_1|}$

- **Box plot (IQR)**: $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$
  - IQR = Q3 - Q1 (intervallo interquartile)

---

# Modelli Parametrici

- **Distribuzione Gaussiana**:
  - Stimare $\mu$ e $\sigma$ dai dati
  - Calcolare la probabilità: $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
  - Anomalie: $p(x) < \theta$ (soglia)

- **Mixture di Gaussiane (GMM)**:
  - Modellare dati multimodali
  - Stimare parametri con EM
  - Anomalie: bassa probabilità sotto il modello

---

# Vantaggi e Svantaggi degli Approcci Statistici

**Vantaggi**:
- Solida base teorica
- Interpretabilità dei risultati
- Efficienza computazionale

**Svantaggi**:
- Assunzioni sulla distribuzione dei dati
- Difficoltà con dati ad alta dimensionalità
- Sensibilità ai parametri di soglia
- Problemi con distribuzioni multimodali (eccetto GMM)

---

# 3. Metodi Basati sulla Densità

- **Idea di base**: Le anomalie si trovano in regioni a bassa densità
- **Approccio**: Stimare la densità locale e identificare come anomalie i punti in regioni a bassa densità

---

# Local Outlier Factor (LOF)

- **Idea**: Confrontare la densità locale di un punto con quella dei suoi vicini
- **Algoritmo**:
  1. Calcolare la distanza al k-esimo vicino per ogni punto
  2. Stimare la densità locale come inverso della distanza media
  3. Calcolare LOF come rapporto tra densità dei vicini e densità del punto
  4. Anomalie: punti con LOF significativamente > 1

![height:250px](https://scikit-learn.org/stable/_images/sphx_glr_plot_lof_outlier_detection_001.png)

---

# DBSCAN per il Rilevamento di Anomalie

- **Approccio**: Utilizzare DBSCAN e considerare i punti di rumore come anomalie
- **Vantaggi**:
  - Identifica cluster di forma arbitraria
  - Naturalmente identifica punti in regioni a bassa densità
  - Non richiede conoscenza a priori del numero di cluster

---

# Vantaggi e Svantaggi dei Metodi Basati sulla Densità

**Vantaggi**:
- Non assumono una distribuzione specifica
- Efficaci con cluster di forme arbitrarie
- Identificano anomalie locali
- Robusti rispetto a diverse densità di cluster

**Svantaggi**:
- Sensibili alla scelta dei parametri (k, ε)
- Computazionalmente costosi per grandi dataset
- Problemi con dati ad alta dimensionalità
- Difficoltà con cluster di densità molto variabile

---

# 4. Isolation Forest

- **Idea di base**: Le anomalie sono più facili da isolare rispetto ai punti normali
- **Principio**: Costruire alberi di decisione casuali e misurare la profondità media necessaria per isolare ogni punto

![height:350px](https://miro.medium.com/max/1400/1*9udWMCjfBBqYRGGJlUAS5Q.png)

---

# Algoritmo Isolation Forest

1. Costruire un insieme di alberi di isolamento:
   - Selezionare casualmente una caratteristica
   - Selezionare casualmente un valore di split tra min e max
   - Ripetere ricorsivamente fino a isolare tutti i punti

2. Calcolare il percorso medio per ogni punto:
   - Anomaly Score: $s(x) = 2^{-\frac{E(h(x))}{c(n)}}$
   - $h(x)$ = lunghezza del percorso
   - $c(n)$ = fattore di normalizzazione

3. Identificare come anomalie i punti con score alto (vicino a 1)

---

# Vantaggi e Svantaggi di Isolation Forest

**Vantaggi**:
- Efficiente: complessità O(n log n)
- Scalabile a grandi dataset
- Robusto ad alta dimensionalità
- Non richiede stime di densità o distanza
- Efficace con anomalie sparse

**Svantaggi**:
- Meno efficace con anomalie collettive
- Sensibile alla presenza di rumore
- Può avere difficoltà con dati fortemente correlati
- Risultati possono variare tra esecuzioni (natura casuale)

---

# 5. One-Class SVM

- **Idea di base**: Trovare un confine che separa la maggior parte dei dati normali dall'origine
- **Approccio**: Mappare i dati in uno spazio ad alta dimensione e trovare l'iperpiano che massimizza la distanza dall'origine

![height:300px](https://scikit-learn.org/stable/_images/sphx_glr_plot_oneclass_001.png)

---

# Algoritmo One-Class SVM

1. Mappare i dati in uno spazio di caratteristiche ad alta dimensione usando un kernel
2. Trovare l'iperpiano che separa i dati dall'origine con il massimo margine
3. La funzione decisionale:
   - $f(x) = \text{sgn}(w^T\phi(x) - \rho)$
   - $w$ = vettore normale all'iperpiano
   - $\rho$ = offset
4. Punti con $f(x) < 0$ sono considerati anomalie

---

# Vantaggi e Svantaggi di One-Class SVM

**Vantaggi**:
- Efficace in spazi ad alta dimensionalità
- Flessibile grazie all'uso di diversi kernel
- Robusto quando ben parametrizzato
- Buone prestazioni con dati non lineari

**Svantaggi**:
- Sensibile alla scelta dei parametri (ν, γ)
- Computazionalmente costoso per grandi dataset
- Difficoltà nel gestire caratteristiche irrilevanti
- Meno interpretabile rispetto ad altri metodi

---

# 6. Rilevamento di Anomalie Basato su Clustering

- **Idea di base**: I punti che non appartengono chiaramente a nessun cluster sono potenziali anomalie
- **Approcci**:
  - Considerare come anomalie i punti non assegnati a cluster
  - Misurare la distanza dai centroidi dei cluster
  - Analizzare la dimensione dei cluster (cluster molto piccoli)

---

# Metodi di Clustering per Anomalie

- **K-means per anomalie**:
  - Eseguire K-means
  - Calcolare la distanza di ogni punto dal centroide più vicino
  - Anomalie: punti con distanza > soglia

- **DBSCAN**:
  - Punti di rumore come anomalie

- **Clustering gerarchico**:
  - Cluster molto piccoli o singleton come anomalie

---

# Vantaggi e Svantaggi del Rilevamento Basato su Clustering

**Vantaggi**:
- Intuitivo e facile da implementare
- Non richiede dati etichettati
- Può scoprire strutture nei dati
- Efficace quando le anomalie formano piccoli cluster

**Svantaggi**:
- Ottimizzato per clustering, non per rilevamento anomalie
- Sensibile ai parametri dell'algoritmo di clustering
- Può non rilevare anomalie all'interno dei cluster
- Computazionalmente costoso per alcuni algoritmi

---

# 7. Rilevamento di Anomalie in Serie Temporali

- **Sfide specifiche**:
  - Dipendenza temporale tra osservazioni
  - Stagionalità e trend
  - Cambiamenti di regime
  - Anomalie di diversa durata (puntuali, collettive)

---

# Tecniche per Serie Temporali

- **Metodi statistici**:
  - Moving Average, ARIMA, Holt-Winters
  - Anomalie: residui oltre una soglia

- **Decomposizione**:
  - Scomporre in trend, stagionalità e residuo
  - Anomalie: residui anomali

- **Reti neurali**:
  - LSTM, GRU, Autoencoders
  - Prevedere il valore successivo
  - Anomalie: errore di previsione elevato

- **Change Point Detection**:
  - Identificare cambiamenti nella distribuzione

---

# Esempio: Rilevamento di Anomalie con LSTM

1. Addestrare una rete LSTM su dati normali per prevedere il valore successivo
2. Calcolare l'errore di previsione per ogni punto
3. Stabilire una soglia basata sulla distribuzione degli errori
4. Identificare come anomalie i punti con errore > soglia

![height:250px](https://miro.medium.com/max/1400/1*Uw_ohxLm7TwCCiNjSKCiDQ.png)

---

# 8. Applicazioni Pratiche del Rilevamento di Anomalie

- **Sicurezza informatica**:
  - Rilevamento di intrusioni
  - Frodi con carte di credito
  - Attività sospette su reti

- **Monitoraggio industriale**:
  - Manutenzione predittiva
  - Controllo qualità
  - Rilevamento guasti

- **Medicina**:
  - Diagnosi di malattie rare
  - Anomalie in immagini mediche
  - Monitoraggio di parametri vitali

---

# Applicazioni Pratiche (cont.)

- **Finanza**:
  - Rilevamento di frodi
  - Trading algoritmico
  - Rischio di credito

- **IoT e Smart Cities**:
  - Monitoraggio del traffico
  - Consumi energetici anomali
  - Qualità dell'aria

- **E-commerce**:
  - Rilevamento di bot
  - Comportamenti di acquisto anomali
  - Recensioni false

---

# Riepilogo: Confronto tra Tecniche di Rilevamento Anomalie

| Tecnica | Complessità | Interpretabilità | Alta dimensionalità | Tipo di anomalie |
|---------|-------------|------------------|---------------------|------------------|
| Statistici | Bassa | Alta | Bassa | Puntuali |
| Basati su densità | Media-Alta | Media | Media | Puntuali, locali |
| Isolation Forest | Bassa | Media | Alta | Puntuali, sparse |
| One-Class SVM | Alta | Bassa | Alta | Puntuali, globali |
| Clustering | Media | Alta | Media | Puntuali, collettive |

---

# Best Practices

- **Preprocessing dei dati**:
  - Normalizzazione/standardizzazione
  - Gestione dei valori mancanti
  - Riduzione della dimensionalità

- **Valutazione**:
  - Precision, Recall, F1-score
  - AUC-ROC, AUC-PR
  - Validazione incrociata

- **Ensemble di metodi**:
  - Combinare diversi approcci
  - Voting o stacking

---

# Domande?

![height:450px](https://miro.medium.com/max/1400/1*JzGt5ItchtwQ9H8aafgQkQ.png)

---

# Riferimenti

- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
- Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers.
- Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). Estimating the support of a high-dimensional distribution.
