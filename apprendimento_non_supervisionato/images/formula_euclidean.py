import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Impostazioni per una migliore qualità dell'immagine
rcParams['font.size'] = 14
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Crea la figura
fig = plt.figure(figsize=(8, 2))
plt.axis('off')

# Aggiungi la formula
plt.text(0.5, 0.5, r'$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$', 
         fontsize=20, ha='center', va='center')

# Salva l'immagine
plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/formula_euclidean.png', 
            bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Distanza di Manhattan
fig = plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.5, 0.5, r'$d(x,y) = \sum_{i=1}^{n}|x_i-y_i|$', 
         fontsize=20, ha='center', va='center')
plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/formula_manhattan.png', 
            bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Distanza di Minkowski
fig = plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.5, 0.5, r'$d(x,y) = \left(\sum_{i=1}^{n}|x_i-y_i|^p\right)^{1/p}$', 
         fontsize=20, ha='center', va='center')
plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/formula_minkowski.png', 
            bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Distanza del coseno
fig = plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.5, 0.5, r'$\cos(\theta) = \frac{x \cdot y}{||x|| \cdot ||y||}$', 
         fontsize=20, ha='center', va='center')
plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/formula_cosine.png', 
            bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Single linkage
fig = plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.5, 0.5, r'$d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$', 
         fontsize=20, ha='center', va='center')
plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/formula_single_linkage.png', 
            bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Complete linkage
fig = plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.5, 0.5, r'$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$', 
         fontsize=20, ha='center', va='center')
plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/formula_complete_linkage.png', 
            bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Average linkage
fig = plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.5, 0.5, r'$d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$', 
         fontsize=20, ha='center', va='center')
plt.savefig('/home/ubuntu/corso_ml_non_supervisionato/images/formula_average_linkage.png', 
            bbox_inches='tight', dpi=300, transparent=True)
plt.close()

print("Immagini delle formule create con successo!")
