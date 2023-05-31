functions {
  real[] diff(real[] x) {
    int N = num_elements(x);
    real result[N-1];
    for (i in 1:(N-1)) {
      result[i] = x[i+1] - x[i];
    }
    return result;
  }
  real[] SEIRS( real t,        // time
				real[] y,      // State S, E, I, R, N por patch 1 and then patch 2
				real[] params, // parameters beta1,beta2,kappa1,kappa2,gama1,gama2
				real[] x_r,    // real ODE argument that depend only on data
				int[] x_i  ){  // integer ODE argument that depend only on data
	
	real S = y[1];
	real E = y[2];
    real I = y[3];
    real R = y[4];
	real Y = y[5];
    real N = y[6];
	
	  real beta   = params[1];
    real kappa  = params[2];
    real gama   = params[3]; 

    real Lambda = x_r[1];
    real mu     = x_r[2];
    real tau    = x_r[3];
	real phi    = x_r[4];

	real dS = Lambda*N - beta*S*I/N - mu*S + tau*R;
    real dE = beta*S*I/N - (kappa + mu) * E;
    real dI = kappa * E - (gama + phi + mu) * I;
    real dR = gama * I - (tau + mu) * R;
    real dY = kappa * E;
    real dN = Lambda*N - (mu * N + phi * I);
    return {dS, dE, dI, dR, dY, dN};
  }
}

data{
  int<lower = 1> n_obs;          // Number of days sampled
  int<lower = 1> n_fake;          // number of fake days sampled for prediction
  int<lower = 0> cases[n_obs]; // The observed data  (daily incidence)
  real<lower = 0> t0;            // Initial time point (zero)
  real<lower = t0> dt[n_obs+1];    // Time points that were sampled
  real<lower = t0> fake_ts[n_fake]; // Fake prediction time points that were sampled
  real<lower = 0> N0;          // Initial pop size 
  real<lower = 0> R0;          // 
  real<lower = 0> Lambda;      // We considere known
  real<lower = 0> mu;          // We considere known
  real<lower = 0> tau;         // We considere known
  real<lower = 0> phi;         // We considere known
}

transformed data {
  real x_r[4] = {Lambda, mu, tau, phi};
  int x_i[0];
}

parameters{                
  //Support of parameters
  real<lower =  0.5, upper = 1.2> beta;
  real<lower =  0.5, upper = 1.2> kappa;
  real<lower =  0.5, upper = 1.2> gama;
  //real<lower =  0.1> nu_inv;
  real<lower =  5e-7> nu;
  real<lower =  0, upper = 15> E0;
  real<lower =  0, upper = 30> I0;
  //real<lower = 5e-7> nu;
}

transformed parameters{
  real <lower = 0.> y[n_obs+1, 6];    // Output from the ODE solver
  real <lower = 0.> y_pred_Inc[n_fake, 6]; // Output from the ODE solver for prediction
  //real nu = 1./nu_inv;
  {
    real y0[6];                 // initial conditions 
    y0[1]   = N0 - ( E0 + I0 + R0);
    y0[2]   = E0;
    y0[3]   = I0;
    y0[4]   = R0; 
	y0[5]   = E0 + I0 + R0;
	y0[6]   = N0;

{
  real params[3]; 
	params[1] = beta;
	params[2] = kappa;
	params[3] = gama;
	//params[4] = nu;
  
    y = integrate_ode_rk45(SEIRS, y0, -0.000001, dt, params, x_r, x_i); //integrate_ode_bdf when stiff
    y_pred_Inc = integrate_ode_rk45(SEIRS, y0, -0.000001, fake_ts, params, x_r, x_i); //integrate_ode_bdf when stiff (for predictions)
  }
}
}

model{  
  // Prior distributions
  beta    ~ normal(0.85, 0.1167); 
  kappa   ~ gamma(53.0816, 62.4490);
  gama    ~ gamma(53.0816, 62.4490);
  //nu_inv ~ gamma(9.0045, 2.2508);
  nu ~ gamma(9.00, 2.25);
  E0      ~ uniform(3.1699, 11.8301);
  I0      ~ uniform(6.3397, 23.6603);
  
  // Likelihood
  
  //for (i in 1:(n_obs)){
  //   cases[i , 1] ~ neg_binomial_2(y[i, 3], phi1);
  //   cases[i , 2] ~ neg_binomial_2(y[i, 8], phi2);
  //}
  cases ~ neg_binomial_2(diff(y[ ,5]), nu);
}
generated quantities {
 
  real pred_Inc_cases[n_fake-1];
  
  pred_Inc_cases = neg_binomial_2_rng(diff(y_pred_Inc[ ,5]), nu);
}





