
### determine hyper parameters of the logNorml distribution for betas
mu = 1
sigma = 1
x1 = 0.74755
x2 = 2.24265
p1 = 0.025
p2 = 0.975
#plnorm(x1, mu, sigma)
#[1] 0.7288829

#plnorm(x2, mu, sigma)
#[1] 0.9036418

errorFn = function(params) (plnorm(x1, params[1], params[2]) - p1)^2 + (plnorm(x2, params[1], params[2]) - p2)^2

result = optim( c(mu, sigma), errorFn)
mean = result$par[1]
sigma = result$par[2]
error = result$value

t = seq(0,5,by=0.01)
w = dlnorm(t,mean, sigma) 
plot(t,w)
abline(v=x1)
abline(v=x2)


### determine hyper parameters of the  gamma distributions for the kappa, gamma and nu parameters
L_kappa = 0.0714
U_kappa = 0.5

L_gamma = 0.0641
U_gamma = 1.5

L_nu = 0.1
U_nu = 10
#pgamma(y1, 1, 1/2)
#[1] 0.7288829

#pgamma(y2, alpha, 1/beta)
#[1] 0.9036418


#errorFn_gamma = function(params) (pgamma(y1, params[1], param[2]) - p1)^2 + (pgamma(x2, params[1], param[2]) - p2)^2

#result = optim( c(alpha, beta), errorFn)

source('~/GammaParmsFromQuantiles.R') # load file where gamma.parms.from.quantiles R function is defined
resul_kappa = gamma.parms.from.quantiles(c(L_kappa,U_kappa), p=c(0.025,0.975), plot=T)
resul_gamma = gamma.parms.from.quantiles(c(L_gamma,U_gamma), p=c(0.025,0.975), plot=T)
resul_nu = gamma.parms.from.quantiles(c(L_nu,U_nu), p=c(0.025,0.975), plot=T)

