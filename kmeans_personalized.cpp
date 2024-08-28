#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Fonction pour calculer les centres de chaque groupe
arma::mat calculerCentres(const arma::mat& data, const arma::ivec& groupes) {
  int maxGroup = groupes.max();
  arma::mat centres(maxGroup, data.n_cols, fill::zeros);

  for (int i = 1; i <= maxGroup; ++i) {
    arma::uvec indices = find(groupes == i);
      arma::mat membresDuGroupe = data.rows(indices);
      centres.row(i - 1) = mean(membresDuGroupe, 0); 
  }

  return centres;
}





// Fonction pour initialiser les centres avec k-means++ utilisant la distance de Mahalanobis
// [[Rcpp::export]]
arma::mat kmeans_plusplus_mahalanobis(const arma::mat& data, int k, const arma::mat& cov_inv, const arma::vec& anomaly_scores) {
  int n = data.n_rows;

  arma::vec inverse_scores = arma::exp(-anomaly_scores);
  arma::vec probabilities = inverse_scores / arma::sum(inverse_scores);
  arma::uvec random_idx_vec = arma::find(arma::randu() < arma::cumsum(probabilities));
  int random_index = random_idx_vec(0);

  arma::mat centers = data.row(random_index);

  if (k == 1) {
    return centers;
  }
  
  for (int i = 1; i < k; ++i) { 
    vec dist_to_centers = zeros<vec>(n);
    
    for (int j = 0; j < n; ++j) {
      vec dists(centers.n_rows);
      
      for (unsigned int m = 0; m < centers.n_rows; ++m) {
        rowvec center = centers.row(m);
        rowvec diff = data.row(j) - center;
        dists(m) = as_scalar(diff * cov_inv * diff.t());
      }
      
      dist_to_centers(j) = min(dists);
    }
    
    NumericVector prob = wrap(dist_to_centers);
    prob = prob / sum(prob);
    IntegerVector sampled_index = sample(n, 1, true, prob); 
    if (sampled_index[0] <= 0 || sampled_index[0] > n) {
      Rcpp::stop("Sampled index out of bounds b");
    }
    centers = join_vert(centers, data.row(sampled_index[0] - 1)); 
  }
  
  return centers;
}


// Fonction pour obtenir la matrice de covariance rétrécie
arma::mat getCovShrink(const arma::mat& data) {
  Environment corpcor("package:corpcor");
  Function cov_shrink = corpcor["cov.shrink"];
  NumericMatrix cov_shrink_result = cov_shrink(Named("x") = data, Named("verbose") = false);
  return as<arma::mat>(cov_shrink_result);
}

// Fonction principale pour k-means utilisant la distance de Mahalanobis
// [[Rcpp::export]]
List kmeansMahalanobis(const DataFrame& df, int k, const arma::vec& anomaly_scores, int iter_max = 50, int n_repeats = 100) {
  int n = df.nrows();
  int p = df.size();
  arma::mat data(n, p);
  
  for (int j = 0; j < p; ++j) {
    data.col(j) = as<arma::vec>(df[j]);
  }
  
  double best_SC = datum::inf;
  arma::mat best_centers;
  arma::ivec best_groupe;
  arma::mat cov_matrix = cov(data);
  double condition_number = cond(cov_matrix);

  if (condition_number > 1e10) {
    cov_matrix = getCovShrink(data); 
  }

  arma::mat cov_inv = inv(cov_matrix);

  if (anomaly_scores.n_elem != n) {
    stop("La taille du vecteur des scores d'anomalie ne correspond pas au nombre de lignes des données.");
  }

  for (int h = 0; h < n_repeats; ++h) {
    arma::mat centers = kmeans_plusplus_mahalanobis(data, k, cov_inv,anomaly_scores);
    arma::ivec groupe(n, fill::zeros);

    for (int iteration = 0; iteration < iter_max; ++iteration) {

      for (int i = 0; i < n; ++i) {
        vec distances = zeros<vec>(k);

        for (unsigned int j = 0; j < centers.n_rows; ++j) {

          rowvec center = centers.row(j);
          rowvec diff = data.row(i) - center;
          distances(j) = as_scalar(diff * cov_inv * diff.t());
        }

        groupe(i) = distances.index_min() + 1; 
      }

      arma::mat new_centers = calculerCentres(data, groupe);

      if (accu(centers != new_centers) == 0) {
        break;
      }

      centers = new_centers;
    }

    double SC_intra_groupe = 0;
    for (int i = 0; i < n; ++i) {

      rowvec center = centers.row(groupe(i) - 1);
      rowvec diff = data.row(i) - center;
      SC_intra_groupe += as_scalar(diff * cov_inv * diff.t());
    }

    if (SC_intra_groupe < best_SC) {
      best_SC = SC_intra_groupe;
      best_centers = centers;
      best_groupe = groupe;
    }
  }

  return List::create(Named("centers") = best_centers,
                      Named("groupe") = best_groupe,
                      Named("SC_intra_groupe") = best_SC);
}
