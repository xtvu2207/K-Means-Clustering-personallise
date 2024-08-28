# K-Means Clustering Personnalisé

Ce code implémente le clustering K-means en utilisant la distance de Mahalanobis, avec une sélection stratégique des centres initiaux pour accélérer la convergence et améliorer la stabilité. Il est implémenté en C++ en utilisant Rcpp pour l'intégration avec R.

### Installation et chargement des dépendances pour le Clustering K-Means Personnalisé

Le code ci-dessous permet d'installer toutes les bibliothèques nécessaires si elles ne sont pas déjà installées sur le système R, puis de les charger automatiquement.

```r
# Fonction pour installer et charger automatiquement les bibliothèques nécessaires
install_and_load_packages = function(packages) {
  new_packages = packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    install.packages(new_packages)
  }
  lapply(packages, require, character.only = TRUE)
}

required_packages = c("Rcpp", "RcppArmadillo", "isotree")

install_and_load_packages(required_packages)
```

Avec ce script, toutes les bibliothèques nécessaires sont installées si elles ne le sont pas déjà, puis chargées automatiquement, en une seule étape.

### Étape suivante : Compilation du fichier C++ et exécution du code

Une fois les bibliothèques installées et chargées, utilisez le code suivant pour compiler le fichier C++ et exécuter la fonction `kmeansMahalanobis` :

```r
# Charger la bibliothèque Rcpp et compiler le fichier C++
library(Rcpp)
sourceCpp("kmeans_personalized.cpp")
```

### Calcul des scores d'anomalie avec Isolation Forest

```r
library(isotree)

# Charger les données numériques et les mettre dans un data frame
data = data_frame_numeric

# Construction du modèle Isolation Forest
iso_model = isolation.forest(data = data, ntrees = 1000, sample_size = nrow(data), ndim = ncol(data))

# Calcul des scores d'anomalie
anomaly_scores = predict(iso_model, data, type = "score")
```

### Utilisation de la fonction `kmeansMahalanobis`

Maintenant que la fonction est chargée, vous pouvez l'utiliser directement dans R. Voici un exemple d'utilisation :

```r
# Nombre de clusters
k = 3

# Appel de la fonction kmeansMahalanobis
result = kmeansMahalanobis(
  df = data,               # Les données sous forme de data frame
  k = k,                   # Le nombre de clusters
  anomaly_scores = anomaly_scores,  # Le vecteur des scores d'anomalie
  iter_max = 50,           # Nombre maximum d'itérations pour le K-means (défaut = 50)
  n_repeats = 100          # Nombre de fois que le K-means est répété pour trouver la meilleure solution (défaut = 100)
)
```

### Interprétation des résultats

La fonction `kmeansMahalanobis` va retourner trois résultats principaux :

1. **`centers`** : Les centres des clusters.
2. **`groupe`** : Les assignations des points aux clusters (vecteur indiquant à quel cluster chaque point appartient).
3. **`SC_intra_groupe`** : La somme des carrés intra-cluster, qui mesure la compacité des clusters.

Vous pouvez afficher ces résultats en utilisant les commandes suivantes :

```r
# Affichage des centres des clusters
print(result$centers)

# Affichage des assignations des points aux clusters
print(result$groupe)

# Affichage de la somme des carrés intra-cluster
print(result$SC_intra_groupe)
```
