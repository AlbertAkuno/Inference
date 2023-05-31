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
	
	real S1 = y[1];
	real E1 = y[2];
    real I1 = y[3];
    real R1 = y[4];
	real Y1 = y[5];
    real N1 = y[6];
	real S2 = y[7];
	real E2 = y[8];
    real I2 = y[9];
    real R2 = y[10];
	real Y2 = y[11];
    real N2 = y[12];
	real S3 = y[13];
	real E3 = y[14];
    real I3 = y[15];
    real R3 = y[16];
	real Y3 = y[17];
    real N3 = y[18];
	real S4 = y[19];
	real E4 = y[20];
    real I4 = y[21];
    real R4 = y[22];
	real Y4 = y[23];
    real N4 = y[24];
	
	real beta1   = params[1];
    real beta2   = params[2];
	real beta3   = params[3];
    real beta4   = params[4];
    real kappa1  = params[5];
    real kappa2  = params[6];
	real kappa3  = params[7];
    real kappa4  = params[8];
    real gama1   = params[9]; 
    real gama2   = params[10];
    real gama3   = params[11]; 
    real gama4   = params[12];

    real Lambda1 = x_r[1];
    real Lambda2 = x_r[2];
	real Lambda3 = x_r[3];
    real Lambda4 = x_r[4];
    real mu1     = x_r[5];
    real mu2     = x_r[6];
	real mu3     = x_r[7];
    real mu4     = x_r[8];
    real tau1    = x_r[9];
    real tau2    = x_r[10];
	real tau3    = x_r[11];
    real tau4    = x_r[12];
	real phi1    = x_r[13];
    real phi2    = x_r[14];
    real phi3    = x_r[15];
    real phi4    = x_r[16];
    real alpha1  = x_r[17];
    real alpha2  = x_r[18];	
	real alpha3  = x_r[19];
    real alpha4  = x_r[20];	
    real p11     = x_r[21];
    real p12     = x_r[22];
	real p13     = x_r[23];
    real p14     = x_r[24];
    real p21     = x_r[25];
    real p22     = x_r[26];
	real p23     = x_r[27];
    real p24     = x_r[28];
	real p31     = x_r[29];
    real p32     = x_r[30];
	real p33     = x_r[31];
    real p34     = x_r[32];
	real p41     = x_r[33];
    real p42     = x_r[34];
	real p43     = x_r[35];
    real p44     = x_r[36];

	real dS1 = Lambda1*N1 - beta1*(1-alpha1) * S1 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4)/((1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4) -
      beta1*alpha1*p11 * S1 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4 )/((1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4 ) -
      beta2*alpha1*p12 * S1 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4 )/((1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) -
      beta3*alpha1*p13 * S1 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/((1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 ) -
      beta4*alpha1*p14 * S1 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/((1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      mu1*S1 + tau1*R1;
    real dE1 = beta1*(1-alpha1) * S1 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4)/((1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4) +
      beta1*alpha1*p11 * S1 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4 )/((1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4 ) +
      beta2*alpha1*p12 * S1 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4 )/((1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) +
      beta3*alpha1*p13 * S1 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/((1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 ) +
      beta4*alpha1*p14 * S1 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/((1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      (kappa1 + mu1) * E1;
    real dI1 = kappa1 * E1 - (gama1 + phi1 + mu1) * I1;
    real dR1 = gama1 * I1 - (tau1 + mu1) * R1;
    real dY1 = kappa1 * E1;
    real dN1 = Lambda1*N1 - (mu1 * N1 + phi1 * I1);
    real dS2 = Lambda2*N2 - beta2*(1-alpha2) * S2 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4)/((1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) -
      beta1*alpha2*p21 * S2 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4 )/((1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4 ) -
      beta2*alpha2*p22 * S2 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4 )/((1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) -
      beta3*alpha2*p23 * S2 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/((1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 ) -
      beta4*alpha2*p24 * S2 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/((1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      mu2*S2 + tau2*R2;
    real dE2 = beta2*(1-alpha2) * S2 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4)/((1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) +
      beta1*alpha2*p21 * S2 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4 )/((1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4 ) +
      beta2*alpha2*p22 * S2 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4 )/((1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) +
      beta3*alpha2*p23 * S2 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/((1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 ) +
      beta4*alpha2*p24 * S2 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/((1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      (kappa2 + mu2) * E2;
    real dI2 = kappa2 * E2 - (gama2 + phi2 + mu2) * I2;
    real dR2 = gama2 * I2 - (tau2 + mu2) * R2;
    real dY2 = kappa2 * E2;
    real dN2 = Lambda2*N2 - (mu2 * N2 + phi2 * I2);
    real dS3 = Lambda3*N3 - beta3*(1-alpha3) * S3 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/((1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3  + alpha4 * p43 * N4 ) -
      beta1*alpha3*p31 * S3 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4 )/((1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4 ) -
      beta2*alpha3*p32 * S3 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4 )/((1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) -
      beta3*alpha3*p33 * S3 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/((1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 ) -
      beta4*alpha3*p34 * S3 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/((1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      mu3*S3 + tau3*R3;
    real dE3 = beta3*(1-alpha3) * S3 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/((1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 ) +
      beta1*alpha3*p31 * S3 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4 )/((1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4 )  +
      beta2*alpha3*p32 * S3 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4 )/((1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 )  +
      beta3*alpha3*p33 * S3 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/((1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 )  +
      beta4*alpha3*p34 * S3 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/((1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 )  -
      (kappa3 + mu3) * E3;
    real dI3 = kappa3 * E3 - (gama3 + phi3 + mu3) * I3;
    real dR3 = gama3 * I3 - (tau3 + mu3) * R3;
    real dY3 = kappa3 * E3;
    real dN3 = Lambda3*N3 - (mu3 * N3 + phi3 * I3);
    real dS4 = Lambda4*N4 - beta4*(1-alpha4) * S4 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/( (1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      beta1*alpha4*p41 * S4 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4 )/( (1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4 ) -
      beta2*alpha4*p42 * S4 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4 )/( (1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) -
      beta3*alpha4*p43 * S4 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/( (1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 ) -
      beta4*alpha4*p44 * S4 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/( (1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      mu4*S4 + tau4*R4;
    real dE4 = beta4*(1-alpha4) * S4 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/( (1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      beta1*alpha4*p41 * S4 * ((1 - alpha1)*I1 + alpha1 * p11 * I1 + alpha2 * p21 * I2 + alpha3 * p31 * I3 + alpha4 * p41 * I4 )/( (1 - alpha1)*N1 + alpha1 * p11 * N1 + alpha2 * p21 * N2 + alpha3 * p31 * N3 + alpha4 * p41 * N4 ) +
      beta2*alpha4*p42 * S4 * ((1 - alpha2)*I2 + alpha1 * p12 * I1 + alpha2 * p22 * I2 + alpha3 * p32 * I3 + alpha4 * p42 * I4 )/( (1 - alpha2)*N2 + alpha1 * p12 * N1 + alpha2 * p22 * N2 + alpha3 * p32 * N3 + alpha4 * p42 * N4 ) +
      beta3*alpha4*p43 * S4 * ((1 - alpha3)*I3 + alpha1 * p13 * I1 + alpha2 * p23 * I2 + alpha3 * p33 * I3 + alpha4 * p43 * I4 )/( (1 - alpha3)*N3 + alpha1 * p13 * N1 + alpha2 * p23 * N2 + alpha3 * p33 * N3 + alpha4 * p43 * N4 ) +
      beta4*alpha4*p44 * S4 * ((1 - alpha4)*I4 + alpha1 * p14 * I1 + alpha2 * p24 * I2 + alpha3 * p34 * I3 + alpha4 * p44 * I4 )/( (1 - alpha4)*N4 + alpha1 * p14 * N1 + alpha2 * p24 * N2 + alpha3 * p34 * N3 + alpha4 * p44 * N4 ) -
      (kappa4 + mu4) * E4;
    real dI4 = kappa4 * E4 - (gama4 + phi4 + mu4) * I4;
    real dR4 = gama4 * I4 - (tau4 + mu4) * R4;
    real dY4 = kappa4 * E4;
    real dN4 = Lambda4*N4 - (mu4 * N4 + phi4 * I4);
    return {dS1, dE1, dI1, dR1, dY1, dN1, dS2, dE2, dI2, dR2, dY2, dN2, dS3, dE3, dI3, dR3, dY3, dN3, dS4, dE4, dI4, dR4, dY4, dN4};
  }
}

data{
  int<lower = 1> n_obs;          // Number of days sampled
  int<lower = 1> n_fake;          // number of fake days sampled for prediction
  int<lower = 0> cases[n_obs,4]; // The observed data  (daily incidence)
  real<lower = 0> t0;            // Initial time point (zero)
  real<lower = t0> dt[n_obs+1];    // Time points that were sampled
  real<lower = t0> fake_ts[n_fake]; // Fake prediction time points that were sampled
  real<lower = 0> N01;          // Initial pop size 
  real<lower = 0> N02;          // Initial pop size 
  real<lower = 0> N03;          // Initial pop size 
  real<lower = 0> N04;          // Initial pop size 
  real<lower = 0> R01;          // 
  real<lower = 0> R02;          // 
  real<lower = 0> R03;          // 
  real<lower = 0> R04;          // 
  real<lower = 0> Lambda1;      // We considere known
  real<lower = 0> Lambda2;      // We considere known
  real<lower = 0> Lambda3;      // We considere known
  real<lower = 0> Lambda4;      // We considere known
  real<lower = 0> mu1;          // We considere known
  real<lower = 0> mu2;          // We considere known 
  real<lower = 0> mu3;          // We considere known
  real<lower = 0> mu4;          // We considere known
  real<lower = 0> tau1;         // We considere known
  real<lower = 0> tau2;         // We considere known   
  real<lower = 0> tau3;         // We considere known
  real<lower = 0> tau4;         // We considere known
  real<lower = 0> phi1;         // We considere known
  real<lower = 0> phi2;         // We considere known
  real<lower = 0> phi3;         // We considere known
  real<lower = 0> phi4;         // We considere known
  real<lower = 0> alpha1;       // We considere known
  real<lower = 0> alpha2;       // We considere known
  real<lower = 0> alpha3;       // We considere known
  real<lower = 0> alpha4;       // We considere known  
  real<lower = 0> p11;          // We considere known
  real<lower = 0> p12;          // We considere known 
  real<lower = 0> p13;          // We considere known
  real<lower = 0> p14;          // We considere known
  real<lower = 0> p21;          // We considere known
  real<lower = 0> p22;          // We considere known 
  real<lower = 0> p23;          // We considere known
  real<lower = 0> p24;          // We considere known
  real<lower = 0> p31;          // We considere known
  real<lower = 0> p32;          // We considere known 
  real<lower = 0> p33;          // We considere known
  real<lower = 0> p34;          // We considere known 
  real<lower = 0> p41;          // We considere known
  real<lower = 0> p42;          // We considere known 
  real<lower = 0> p43;          // We considere known
  real<lower = 0> p44;          // We considere known   
}

transformed data {
  real x_r[36] = {Lambda1, Lambda2, Lambda3, Lambda4, mu1, mu2, mu3, mu4, tau1, tau2, tau3, tau4, phi1, phi2, phi3, phi4, alpha1, alpha2, alpha3, alpha4, p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, p41, p42, p43, p44};
  int x_i[0];
}

parameters{                
  //Support of parameters
  real<lower =  0.9, upper = 1.3> beta1;
  real<lower =  0.8, upper = 1.2> beta2;
  real<lower =  0.3, upper = 0.6> beta3;
  real<lower =  1.3, upper = 1.68> beta4;
  real<lower =  0.2, upper = 0.4> kappa1;
  real<lower =  0.2, upper = 0.4> kappa2;
  real<lower =  0.2, upper = 0.4> kappa3;
  real<lower =  0.2, upper = 0.4> kappa4;
  real<lower =  0.8, upper = 1.2> gamma1;
  real<lower =  0.8, upper = 1.2> gamma2;
  real<lower =  0.8, upper = 1.2> gamma3;
  real<lower =  0.7, upper = 0.9> gamma4;
  real<lower =  0.1> nu1_inv;
  real<lower =  0.1> nu2_inv;
  real<lower =  0.1> nu3_inv;
  real<lower =  0.1> nu4_inv;
  real<lower =  50, upper = 65> E01;
  real<lower =  70, upper = 85> E02;
  real<lower =  15, upper = 26> E03;
  real<lower =  2, upper = 7> E04;
  real<lower =  0, upper = 10> I01;
  real<lower =  0, upper = 10> I02;
  real<lower =  0, upper = 10> I03;
  real<lower =  0, upper = 10> I04;
}

transformed parameters{
  real <lower = 0.> y[n_obs+1, 24];    // Output from the ODE solver
  real <lower = 0.> y_pred_Inc[n_fake, 24]; // Output from the ODE solver for prediction
  real nu1 = 1./nu1_inv;
  real nu2 = 1./nu2_inv;
  real nu3 = 1./nu3_inv;
  real nu4 = 1./nu4_inv;
  {
    real y0[24];                 // initial conditions 
    y0[1]   = N01 - ( E01 + I01 + R01);
    y0[2]   = E01;
    y0[3]   = I01;
    y0[4]   = R01; 
	y0[5]   = E01 + I01 + R01;
	y0[6]   = N01;
    y0[7]   = N02 - ( E02 + I02 + R02);
    y0[8]   = E02;
    y0[9]   = I02;
    y0[10]  = R02; 
	y0[11]  = E02 + I02 + R02;
	y0[12]  = N02;
    y0[13]  = N03 - ( E03 + I03 + R03);
    y0[14]  = E03;
    y0[15]  = I03;
    y0[16]  = R03; 
	y0[17]  = E03 + I03 + R03;
	y0[18]  = N03;
    y0[19]  = N04 - ( E04 + I04 + R04);
    y0[20]  = E04;
    y0[21]  = I04;
    y0[22]  = R04; 
	y0[23]  = E04 + I04 + R04;
	y0[24]  = N04;

{
  real params[12]; 
	params[1] = beta1;
	params[2] = beta2;
	params[3] = beta3;
	params[4] = beta4;
	params[5] = kappa1;
	params[6] = kappa2;
	params[7] = kappa3;
	params[8] = kappa4;
	params[9] = gamma1;
	params[10] = gamma2;
	params[11] = gamma3;
	params[12] = gamma4;
  
    y = integrate_ode_rk45(SEIRS, y0, -0.000001, dt, params, x_r, x_i); //integrate_ode_bdf when stiff
    y_pred_Inc = integrate_ode_rk45(SEIRS, y0, -0.000001, fake_ts, params, x_r, x_i); //integrate_ode_bdf when stiff (for predictions)
  }
}
}

model{  
  // Prior distributions
  beta1    ~ lognormal(0.4, 0.05); 
  beta2    ~ lognormal(0.4, 0.05);
  beta3    ~ lognormal(0.4, 0.05); 
  beta4    ~ lognormal(0.4, 0.05);
  kappa1   ~ gamma(1.5, 0.5);
  kappa2   ~ gamma(1.5, 0.5);
  kappa3   ~ gamma(1.5, 0.5);
  kappa4   ~ gamma(1.5, 0.5);  
  gamma1    ~ gamma(1.5, 0.5); 
  gamma2    ~ gamma(1.5, 0.5); 
  gamma3    ~ gamma(1.5, 0.5); 
  gamma4    ~ gamma(1.5, 0.5); 
  nu1_inv ~ inv_gamma(0.2, 0.5);
  nu2_inv ~ inv_gamma(0.2, 0.5);
  nu3_inv ~ inv_gamma(0.2, 0.5);
  nu4_inv ~ inv_gamma(0.2, 0.5);
  E01      ~ uniform(50., 65);
  E02      ~ uniform(70., 85);
  E03      ~ uniform(15., 26);
  E04      ~ uniform(2., 7);
  I01      ~ uniform(0., 20);
  I02      ~ uniform(0., 20);
  I03      ~ uniform(0., 20);
  I04      ~ uniform(0., 20);
  
  // Likelihood
  
  //for (i in 1:(n_obs)){
  //   cases[i , 1] ~ neg_binomial_2(y[i, 3], phi1);
  //   cases[i , 2] ~ neg_binomial_2(y[i, 8], phi2);
  //}
  cases[ , 1] ~ neg_binomial_2(diff(y[ ,5]), nu1);
  cases[ , 2] ~ neg_binomial_2(diff(y[ ,11]), nu2);
  cases[ , 3] ~ neg_binomial_2(diff(y[ ,17]), nu3);
  cases[ , 4] ~ neg_binomial_2(diff(y[ ,23]), nu4);
}
generated quantities {
 
  real pred_Inc_cases1[n_fake-1];
  real pred_Inc_cases2[n_fake-1];
  real pred_Inc_cases3[n_fake-1];
  real pred_Inc_cases4[n_fake-1];
  
  pred_Inc_cases1 = neg_binomial_2_rng(diff(y_pred_Inc[ ,5]), nu1);
  pred_Inc_cases2 = neg_binomial_2_rng(diff(y_pred_Inc[ ,11]), nu2);
  pred_Inc_cases3 = neg_binomial_2_rng(diff(y_pred_Inc[ ,17]), nu3);
  pred_Inc_cases4 = neg_binomial_2_rng(diff(y_pred_Inc[ ,23]), nu4);
}



