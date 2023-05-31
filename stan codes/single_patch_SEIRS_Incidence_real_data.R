setwd("~/Library/CloudStorage/Dropbox/Inference on th Multi-patch Epidemic model_Bayesian UQ/stan_four_regions/single_patch_incidence_real_data")
library(deSolve)
library(ggplot2)
library(ggpubr)
library(rstan)
library(gridExtra)
library(Rcpp)
library("RcppCNPy")
library(reticulate) #for reading numpy files
np <- import("numpy") #load numpy (you will have to install miniconda)
rstan_options (auto_write = TRUE)
options (mc.cores = parallel::detectCores ())

global_cases = np$load("/Volumes/F/Hermosillo_four_regions data/global_COVID-19_data/global_COVID-19_data.npy")

hermosillo_global_COVID_including_pred_data = np$load("/Volumes/F/Hermosillo_four_regions data/global_COVID-19_data/global_COVID-19_including_pred_data.npy")

#zone2_cases = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone2_COVID-19_data.npy")
#zone3_cases = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone3_COVID-19_data.npy")
#zone4_cases = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone4_COVID-19_data.npy")

global_cases_inflated = global_cases*15
global_including_pred_cases_inflated = hermosillo_global_COVID_including_pred_data*15


#zone2_cases_inflated = zone2_cases*15
#zone3_cases_inflated = zone3_cases*15
#zone4_cases_inflated = zone4_cases*15


#set.seed(3)
yr1 = global_cases_inflated
yr2 = global_including_pred_cases_inflated

#yr = cbind(zone1_cases_inflated, zone2_cases_inflated,
#           zone3_cases_inflated, zone4_cases_inflated)

#colnames(yr) = c("Obs_Inc1")

#dt <- seq(0,length(zone4_cases),1)

dt <- seq(0,length(global_cases),1)
# Plotting noisy simulated data ----

#par(mfrow=c(2,2))
plot(dt[-1],global_cases_inflated,type="p",pch=20,col="blue", xlab = "time", ylab = "Count")
title(main="Patch 1")
#legend("topright", legend=c("Patch 1","Patch 2","Patch 3","Patch 4"), col=c("red","blue","orange", "darkgreen"), 
       #lty=c(1,1,1,1,2,2),lwd=2)


# Running Stan ----

ts  = seq(0.0,length(yr1)-1,1)
times_pred = seq(0.0,230,1)
#times_pred = seq(0.0,60,1)
dt = seq(0.0,length(yr1),1)
n_obs = length(ts)
#n_obs = length(times_pred)
#n_sample = length(ts)
n_fake = length(times_pred)
t0 = 0.0#-0.0000001
fake_ts = times_pred

#Initial conditions
R0  = 0              # Removed number of individuals
N01  = 231735          # Initial population sizes
N02  = 295365
N03  = 183204
N04  = 148508
N0 = N01 + N02 + N03 + N04
Lambda = 15.7/(1000*365)
mu     = 0.06/(1000*365)
tau    = 1/180
phi    = 0.00385


#I pass dt instead of ts as it is one term longer than ts and when we take the differences 
#of consecutive terms in Y's, we end with vector with the same lengths as the data.
data_SEIRS <- list(n_obs   = n_obs, 
                   n_fake  = n_fake,
                   cases   = yr1,   #incidences
                   t0      = t0, 
                   dt      = dt,
                   fake_ts = fake_ts,
                   N0     = N0,   
                   R0     = R0, 
                   Lambda = Lambda,
                   mu     = mu,
                   tau    = tau,
                   phi    = phi)

#stanc("Multipatch4_SEIRS_incidence.stan")

model <- stan_model("single_patch_SEIRS_incidence_real_data_server.stan")

n_chains = 2 #4
n_warmups = 20000
n_iter = 40000
n_thin = 100
set.seed(1234)

init_fun0 = function(){
  list(beta    = 0.6,   #beta
       kappa   = 0.6,    #kappa
       gama    = 0.6,  #gamma
       nu = 0.3,    #nu_inv
       E0      = 7, 
       I0      = 15
  )
}



init_fun = function(){
  list(beta    = runif(1, 0.5, 1.0),   #beta
       kappa   = runif(1, 0.5, 1.0),    #kappa
       gama    = runif(1, 0.5, 0.9),  #gamma
       nu_inv = runif(1, 0.0, 0.3),    #nu_inv
       E0      = runif(1,0,200), 
       I0      = runif(1,0,100)
   )
}


fit_SEIR <- sampling(model,
                     data   = data_SEIRS,
                     chains = n_chains, 
                     warmup = n_warmups,
                     iter   = n_iter,
                     #init   = init_fun0,
                     thin   = n_thin,
                     seed   = 5,
                     control=list(adapt_delta=0.99, stepsize = 0.01, max_treedepth =15))


# Saving an viewing Stan results ----

#save(fit_SEIR, file = "fit_SEIR.RData")

load(file = "fit_SEIRS_single_patch.RData")

pars = c("beta","kappa","gama","E0","I0","nu")

print(fit_SEIRS_single_patch, pars = pars, digits_summary = 4)




stan_dens(fit_SEIRS_single_patch, pars = pars, separate_chains = TRUE)
ggsave("posterior_distributions_beta_kappa_gamma_E0_I0.png", width = 30, height = 17, units = c("cm"), dpi=300)

pairs(fit_SEIRS_single_patch, pars = pars)
#ggsave("pairs_beta_kappa_gamma.pdf", width = 30, height = 30, units = c("cm"), dpi=300)

stan_trace(fit_SEIRS_single_patch, pars = pars)
ggsave("traceplots_beta_kappa_gamma_E0_I0.png", width = 30, height = 17, units = c("cm"), dpi=300)

stan_plot(fit_SEIRS_single_patch, pars = pars )
ggsave("credible_intervals_beta_kappa_gamma_E0_I0.png", width = 20, height = 17, units = c("cm"), dpi=300)


list_of_draws <- extract(fit_SEIRS_single_patch)
print(names(list_of_draws))

tail(list_of_draws$I0)



start_date = "2020-02-26"
end_date = "2020-09-06"

pred_date = "2020-10-17"


dates_pred <- seq(as.Date('2020-02-26'), as.Date('2020-10-17'), by = 'days')

#dates_data <- seq(as.Date('2017-01-01'), as.Date('2017-12-31'), by = 'days')

# Checking  ----
yr_1 = data.frame(yr1)
yr_2 = data.frame(yr2)

smr_pred <- cbind(as.data.frame(summary(fit_SEIRS_single_patch, pars = "pred_Inc_cases", probs = c(0.05, 0.5, 0.95))$summary), dates_pred[6:235])
colnames(smr_pred) <- make.names(colnames(smr_pred)) # to remove % in the col names
ggplot(smr_pred, mapping = aes(x = dates_pred[6:235])) +
  geom_ribbon(aes(ymin = X5., ymax = X95., fill = "95% CI"), alpha = 0.35) +
  geom_line(mapping = aes(x = dates_pred[6:235], y = X50., color = "brown4")) + 
  geom_line(mapping = aes(x = dates_pred[6:235], y = mean, color = "blue")) + 
  geom_point(data=yr_1, mapping = aes(x=dates_pred[6:193], y = yr1, color = "black"), size=0.8) +
  geom_point(data=yr_2, mapping = aes(x=dates_pred[6:234], y = yr2, color = "black"), size=0.8) +
  geom_vline(data=yr_1, aes(xintercept=as.numeric(as.Date('2020-09-07')), colour="darkgreen"), linetype="solid", lwd=0.8, show.legend = F)+
  scale_x_date(date_breaks = "1 month", date_labels = "%Y-%m")+
  annotate(x = as.Date('2020-09-07'), y = +Inf, label = "2020-09-07", vjust = 2, geom = "label") +
  scale_color_identity(name = "Predictive check:",
                       breaks = c("gold2", "brown4", "blue", "black", "darkgreen"),
                       labels = c("95% CI", "Median model incidence", "Mean model incidence", "Observed daily incidence", "Prediction start date"),
                       guide = "legend")+
  theme(legend.position = c(0.2, 0.75), 
        legend.background = element_rect(fill = "white", color = "black"), 
        legend.title = element_text(face = "bold"),plot.title = element_text(hjust = 0.5))+
  ggtitle(expression("Global")) +
  labs(x = "Time", y = "Incidence")

ggsave("single_patch_predictive_checks.png", width = 20, height = 15, units = c("cm"), dpi=300)

predicted_mean_global_incidence = smr_pred$mean[194:229]
predicted_median_global_incidence = smr_pred$X50.[194:229]

observed_global_prediction_incidence = yr2[194:229]
n= length(predicted_mean_global_incidence)

mean_global_squared_deviation = (predicted_mean_global_incidence - observed_global_prediction_incidence)^2

sum_mean_global_squared_deviation = sum(mean_global_squared_deviation)

MSE_global_mean = sqrt(sum_mean_global_squared_deviation/n)

mean_global_relative_difference = abs( (observed_global_prediction_incidence -predicted_mean_global_incidence)/observed_global_prediction_incidence   )
sum_mean_global_relative_difference = sum(mean_global_relative_difference)
MAPE_mean_global = (sum_mean_global_relative_difference/n)*100


median_global_squared_deviation = (predicted_median_global_incidence - observed_global_prediction_incidence)^2

sum_median_global_squared_deviation = sum(median_global_squared_deviation)

MSE_global_median = sqrt(sum_median_global_squared_deviation/n)

median_global_relative_difference = abs( (observed_global_prediction_incidence -predicted_median_global_incidence)/observed_global_prediction_incidence   )
sum_median_global_relative_difference = sum(median_global_relative_difference)
MAPE_median_global = (sum_median_global_relative_difference/n)*100

