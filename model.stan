// Hierarchical Beta regression for MUSHRA scores

data {
  int<lower=1> N;                      // total number of observations
  int<lower=1> S;                      // number of systems tested
  array[N] int<lower=0> system_id;     //system id of each observation
  
  array[N] real<lower=0, upper=1> y;   //Given scores(rescaled 0 to 1)
}

parameters {
  real mu_0;                           // global intercept on logit scale
  array[S] real alpha_j;               // system offsets (logit scale)
  real<lower=0> sigma_alpha;           // sd for system effects
  array[S] real<lower=0> phi_j;        // Precision of the beta distributions
}

transformed parameters {
  // Eta is the linear predictor on logit scale for each observation
  vector[N] eta;
  for (n in 1:N) {
    eta[n] = mu_0 + alpha_j[system_id[n]];
  }
}

model{
  // Prior definition
  mu_0 ~ normal(0,1);
  alpha_j ~ normal(0,sigma_alpha);
  sigma_alpha ~ inv_gamma(3,1);
  phi_j ~ gamma(2, 0.1);
  
  // Likelihood: parametrize Beta by mean mu (inv_logit(eta)) and the precission
  for(n in 1:N){
    real mu_j = inv_logit(eta[n]);
    y[n] ~ beta(mu_j * phi_j[system_id[n]], (1-mu_j) * phi_j[system_id[n]]);
  }
}
