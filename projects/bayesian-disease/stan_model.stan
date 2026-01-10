// Code for ODE
functions {
  vector sir(real t, vector y, array[] real theta, 
             array[] real x_r, array[] int x_i) {

      real S = y[1];
      real I = y[2];
      real R = y[3];
      real N = x_i[1];
      
      real beta = theta[1];
      real gamma = theta[2];
      
      real dS_dt = -beta * I * S / N;
      real dI_dt =  beta * I * S / N - gamma * I;
      real dR_dt =  gamma * I;
      
      // ' transposes our row vector into a column vector
      return [dS_dt, dI_dt, dR_dt]';
  }
}

data {
  int<lower=1> n_days;
  vector[3] y0;
  array[n_days] real t;
  int N;
  array [n_days] int cases;
}

// Track status of individuals in SIR model in x
transformed data {
  real t0 = 0; 
  array[0] real x_r;
  array[1] int x_i = { N };
}

// Truncated prior using <lower=...>
parameters {
  real<lower=0> beta;
  real<lower=0> gamma;
  real<lower=0> phi_inv_sqrt;
}

transformed parameters{
  array[n_days] vector[3] y;
  real phi = 1. / (square(phi_inv_sqrt));
  {
    array[2] real theta;
    theta[1] = beta;
    theta[2] = gamma;

    y = ode_rk45(sir, y0, t0, t, theta, x_r, x_i);
  }
}

model {
  //priors 
  beta ~ normal(1.7, 1.4); 
  gamma ~ normal(0.2, 0.53);
  phi_inv_sqrt ~ normal(0, 1);
  
  // sampling distribution
  cases ~ neg_binomial_2(y[,2], phi);
}

// Quantities of interest
generated quantities {
  real R0 = beta / gamma;
  real recovery_time = 1 / gamma;
  array[n_days] real pred_cases;
  pred_cases = neg_binomial_2_rng(y[,2], phi);
}

