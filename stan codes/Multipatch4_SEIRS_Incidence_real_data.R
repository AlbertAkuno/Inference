setwd("~/Library/CloudStorage/Dropbox/Inference on th Multi-patch Epidemic model_Bayesian UQ/stan_four_regions/Multipatch4_SEIRS_Incidence_real data")
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

zone1_cases = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone1_COVID-19_data.npy")
zone2_cases = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone2_COVID-19_data.npy")
zone3_cases = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone3_COVID-19_data.npy")
zone4_cases = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone4_COVID-19_data.npy")

zone1_cases_including_pred_data = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone1_COVID-19_including_pred_data.npy")
zone2_cases_including_pred_data = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone2_COVID-19_including_pred_data.npy")
zone3_cases_including_pred_data = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone3_COVID-19_including_pred_data.npy")
zone4_cases_including_pred_data = np$load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone4_COVID-19_including_pred_data.npy")

zone1_cases_inflated = zone1_cases*15
zone2_cases_inflated = zone2_cases*15
zone3_cases_inflated = zone3_cases*15
zone4_cases_inflated = zone4_cases*15

zone1_cases_including_pred_data_inflated = zone1_cases_including_pred_data*15
zone2_cases_including_pred_data_inflated = zone2_cases_including_pred_data*15
zone3_cases_including_pred_data_inflated = zone3_cases_including_pred_data*15
zone4_cases_including_pred_data_inflated = zone4_cases_including_pred_data*15

#set.seed(3)
yr = cbind(zone1_cases_inflated, zone2_cases_inflated,
           zone3_cases_inflated, zone4_cases_inflated)

yr2_zones_pred_data = cbind(zone1_cases_including_pred_data_inflated, zone2_cases_including_pred_data_inflated,
                            zone3_cases_including_pred_data_inflated, zone4_cases_including_pred_data_inflated)

colnames(yr) = c("Obs_Inc1", "Obs_Inc2","Obs_Inc3","Obs_Inc4")

dt <- seq(0,length(zone4_cases),1)
# Plotting noisy simulated data ----

par(mfrow=c(2,2))
plot(dt[-1],zone1_cases_inflated,type="p",pch=20,col="blue", xlab = "time", ylab = "Count")
title(main="Patch 1")
plot(dt[-1],zone2_cases_inflated,type="p",pch=20,col="blue", xlab = "time", ylab = "Count")
title(main="Patch 2")
plot(dt[-1],zone3_cases_inflated,type="p",pch=20,col="blue", xlab = "time", ylab = "Count")
title(main="Patch 3")
plot(dt[-1],zone4_cases_inflated,type="p",pch=20,col="blue", xlab = "time", ylab = "Count")
title(main="Patch 4")
#legend("topright", legend=c("Patch 1","Patch 2","Patch 3","Patch 4"), col=c("red","blue","orange", "darkgreen"), 
       #lty=c(1,1,1,1,2,2),lwd=2)


# Running Stan ----

ts  = seq(0.0,length(yr[,1])-1,1)
times_pred = seq(0.0,230,1)
#times_pred = seq(0.0,60,1)
dt = seq(0.0,length(yr[,1]),1)
n_obs = length(ts)
#n_obs = length(times_pred)
#n_sample = length(ts)
n_fake = length(times_pred)
t0 = 0.0#-0.0000001
fake_ts = times_pred

#Initial conditions
R01  = 0              # Removed number of individuals
R02  = 0
R03  = 0
R04  = 0
N01  = 231735          # Initial population sizes
N02  = 295365
N03  = 183204
N04  = 148508
Lambda1 = 15.7/(1000*365)
Lambda2 = 15.7/(1000*365)
Lambda3 = 15.7/(1000*365)
Lambda4 = 15.7/(1000*365)
mu1     = 0.06/(1000*365)
mu2     = 0.06/(1000*365)
mu3     = 0.06/(1000*365)
mu4     = 0.06/(1000*365)
tau1    = 1/180
tau2    = 1/180
tau3    = 1/180
tau4    = 1/180
phi1    = 0.00385
phi2    = 0.00385
phi3    = 0.00385
phi4    = 0.00385
alpha1  = 0.9668
alpha2  = 0.9265
alpha3  = 0.9692
alpha4  = 0.9680
p11     = 0.8164
p12     = 0.1289 
p13     = 0.0372
p14     = 0.0175
p21     = 0.1222
p22     = 0.8119 
p23     = 0.0215
p24     = 0.0444
p31     = 0.0722
p32     = 0.0504
p33     = 0.7293
p34     = 0.1481
p41     = 0.0313
p42     = 0.1166
p43     = 0.1278
p44     = 0.7243

#I pass dt instead of ts as it is one term longer than ts and when we take the differences 
#of consecutive terms in Y's, we end with vector with the same lengths as the data.
data_SEIRS <- list(n_obs   = n_obs, 
                   n_fake  = n_fake,
                   cases   = yr,   #incidences
                   t0      = t0, 
                   dt      = dt,
                   fake_ts = fake_ts,
                   N01     = N01,   
                   N02     = N02,
                   N03     = N03, 
                   N04     = N04,
                   R01     = R01, 
                   R02     = R02,
                   R03     = R03, 
                   R04     = R04,
                   Lambda1 = Lambda1,
                   Lambda2 = Lambda2,
                   Lambda3 = Lambda3,
                   Lambda4 = Lambda4,
                   mu1     = mu1,
                   mu2     = mu2,
                   mu3     = mu3,
                   mu4     = mu4,
                   tau1    = tau1,
                   tau2    = tau2,
                   tau3    = tau3,
                   tau4    = tau4,
                   phi1    = phi1,
                   phi2    = phi2,
                   phi3    = phi3,
                   phi4    = phi4,
                   alpha1  = alpha1,
                   alpha2  = alpha2,
                   alpha3  = alpha3,
                   alpha4  = alpha4,
                   p11     = p11,
                   p12     = p12,
                   p13     = p13,
                   p14     = p14,
                   p21     = p21,
                   p22     = p22,
                   p23     = p23,
                   p24     = p24,
                   p31     = p31,
                   p32     = p32,
                   p33     = p33,
                   p34     = p34,
                   p41     = p41,
                   p42     = p42,
                   p43     = p43,
                   p44     = p44)

#stanc("Multipatch4_SEIRS_incidence.stan")

model <- stan_model("Multipatch4_SEIRS_incidence_real_data_server.stan")

n_chains = 1 #4
n_warmups = 300
n_iter = 1000
n_thin = 10
set.seed(1234)

init_fun0 = function(){
  list(beta1    = 1.0950,   #beta
       beta2    = 1.0473,
       beta3    = 0.4904,
       beta4    = 1.4951,
       kappa1   = 0.3,    #kappa
       kappa2   = 0.3,
       kappa3   = 0.3,
       kappa4   = 0.3,
       gama1    = 1.0,  #gamma
       gama2    = 1.0,
       gama3    = 1.0,
       gama4    = 1.0,
       nu1      = 8.9,    #phi_inv
       nu2      = 8.2,
       nu3      = 4.6,
       nu4      = 5.4,
       E01      = 58.9925,
       E02      = 78.1313,
       E03      = 22.9404,
       E04      = 4.4192,
       I01      = 6.3662,
       I02      = 5.5063,
       I03      = 5.7946,
       I04      = 2.6886
  )
}



init_fun = function(){
  list(beta1    = runif(1, 0.9, 1.3),   #beta
       beta2    = runif(1, 0.8, 1.2), 
       beta3    = runif(1, 0.3, 0.6), 
       beta4    = runif(1, 1.3, 1.68), 
       kappa1   = runif(1, 0.2, 0.4),    #kappa
       kappa2   = runif(1, 0.2, 0.4), 
       kappa3   = runif(1, 0.2, 0.4),  
       kappa4   = runif(1, 0.2, 0.4), 
       gama1    = runif(1, 0.8, 1.2),  #gamma
       gama2    = runif(1, 0.8, 1.2), 
       gama3    = runif(1, 0.8, 1.2),
       gama4    = runif(1, 0.7, 0.9), 
       phi1_inv = runif(1, 0.0, 0.3),    #phi_inv
       phi2_inv = runif(1, 0.0, 0.3),
       phi3_inv = runif(1, 0.0, 0.3), 
       phi4_inv = runif(1, 0.0, 0.3),
       E01      = runif(1,50,65), 
       E02      = runif(1,70,85), 
       E03      = runif(1,15,26), 
       E04      = runif(1,2,7), 
       I01      = runif(1,1,10), 
       I02      = runif(1,1,10),
       I03      = runif(1,1,10), 
       I04      = runif(1,1,10) 
   )
}


fit_SEIR <- sampling(model,
                     data   = data_SEIRS,
                     chains = n_chains, 
                     warmup = n_warmups,
                     iter   = n_iter,
                     init   = init_fun0,
                     thin   = n_thin,
                     seed   = 5,
                     control=list(adapt_delta=0.95, stepsize = 0.05, max_treedepth = 12))

extractted_chains = extract(fit_SEIR)

head(extractted_chains)

plot(seq(1:8), extractted_chains[1:24])
# Saving an viewing Stan results ----

#save(fit_SEIR, file = "fit_SEIR.RData")

load(file = "fit_SEIRS.RData")

#load(file = "final_fit_SEIRS.RData")



epid_pars=c("beta1","beta2","beta3","beta4","kappa1","kappa2","kappa3","kappa4", "gamma1","gamma2","gamma3","gamma4" )

epid_pars_beta_gamma=c("beta1","beta2","beta3","beta4","gamma1","gamma2","gamma3","gamma4" )


init_nu_pars = c("E01","E02","E03","E04","I01","I02", "I03","I04", "nu1", "nu2","nu3", "nu4" )

print(fit_SEIRS, pars = epid_pars, digits_summary = 4)
print(fit_SEIRS, pars = init_nu_pars, digits_summary = 4)



stan_dens(fit_SEIRS, pars = epid_pars, separate_chains = TRUE, fill = "blue")
ggsave("posterior_distributions_beta_kappa_gamma.png", width = 15, height = 10, units = c("cm"), dpi=300)

stan_dens(fit_SEIRS, pars = init_nu_pars, separate_chains = TRUE, fill="blue")
ggsave("posterior_distributions_E0_I0_nu.png", width = 15, height = 10, units = c("cm"), dpi=300)

#stan_dens(fit_SEIRS, pars = pars4, separate_chains = TRUE)
#stan_dens(fit_SEIRS, pars = pars1, separate_chains = TRUE)
#stan_dens(fit_SEIRS, pars = pars5, separate_chains = TRUE)
#stan_dens(fit_SEIRS, pars = pars6, separate_chains = TRUE)

pairs(fit_SEIRS, pars = epid_pars_beta_gamma)



pairs(fit_SEIRS, pars = epid_pars)
#ggsave("pairs_beta_kappa_gamma.pdf", width = 30, height = 30, units = c("cm"), dpi=300)

pairs(fit_SEIRS, pars = init_nu_pars)

#pairs(fit_SEIRS, pars = epid_pars2)
#ggsave("pairs_gamma_E0.png", width = 30, height = 17, units = c("cm"), dpi=300)

#pairs(fit_SEIRS, pars = epid_pars3)
#ggsave("pairs_I0_nu.png", width = 30, height = 17, units = c("cm"), dpi=300)

#pairs(fit_SEIRS, pars = init_nu_pars)

#stan_trace(fit_SEIRS, pars = pars3)

stan_trace(fit_SEIRS, pars = epid_pars, color="blue")
ggsave("traceplots_beta_kappa_gamma_stan.png", width = 15, height = 10, units = c("cm"), dpi=300)

stan_trace(fit_SEIRS, pars = init_nu_pars, color="blue")
ggsave("traceplots_E0_I0_nu_stan.png", width = 15, height = 10, units = c("cm"), dpi=300)

stan_plot(fit_SEIRS, pars = epid_pars )
ggsave("credible_intervals_beta_kappa_gamma_stan.png", width = 15, height = 10, units = c("cm"), dpi=300)

stan_plot(fit_SEIRS, pars = init_nu_pars )
ggsave("credible_intervals_E0_I0_nu.png", width = 20, height = 17, units = c("cm"), dpi=300)

m<-stan_plot(fit_SEIRS, pars = epid_pars )
n<-stan_plot(fit_SEIRS, pars = init_nu_pars )
ggarrange(m,n, labels = c("A", "B"), ncol = 2, nrow = 1)
ggsave("credible_intervals_beta_kappa_gamma_E0_I0_nu_stan.png", width = 15, height = 10, units = c("cm"), dpi=300)




start_date = "2020-02-26"
end_date = "2020-09-06"

pred_date = "2020-10-17"


dates_pred <- seq(as.Date('2020-02-26'), as.Date('2020-10-17'), by = 'days')

#dates_data <- seq(as.Date('2017-01-01'), as.Date('2017-12-31'), by = 'days')

# Checking  ----
yr1 = data.frame(yr)

yr_2_zones_pred_data = data.frame(yr2_zones_pred_data)

smr_pred1 <- cbind(as.data.frame(summary(fit_SEIRS, pars = "pred_Inc_cases1", probs = c(0.05, 0.5, 0.95))$summary), dates_pred[6:235])
colnames(smr_pred1) <- make.names(colnames(smr_pred1)) # to remove % in the col names
p<-ggplot(smr_pred1, mapping = aes(x = dates_pred[6:235])) +
  geom_ribbon(aes(ymin = X5., ymax = X95., fill = "95% CI"), alpha = 0.35) +
  geom_line(mapping = aes(x = dates_pred[6:235], y = X50., color = "brown4")) + 
  geom_line(mapping = aes(x = dates_pred[6:235], y = mean, color = "blue")) + 
  geom_point(data=yr1, mapping = aes(x=dates_pred[6:193], y = yr[,1], color = "black"), size=0.8) +
  geom_point(data=yr_2_zones_pred_data, mapping = aes(x=dates_pred[6:234], y = yr2_zones_pred_data[,1], color = "black"), size=0.8) +
  geom_vline(data=yr1, aes(xintercept=as.numeric(as.Date('2020-09-07')), colour="darkgreen"), linetype="solid", lwd=0.8, show.legend = F)+
  scale_x_date(date_breaks = "1 month", date_labels = "%Y-%m")+
  annotate(x = as.Date('2020-09-07'), y = +Inf, label = "2020-09-07", vjust = 2, geom = "label") +
  scale_color_identity(name = "Predictive check:",
                       breaks = c("gold2", "brown4", "blue", "black", "darkgreen"),
                       labels = c("95% CI", "Median model incidence", "Mean model incidence", "Observed daily incidence", "Prediction start date"),
                       guide = "legend")+
  #theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.position = c(0.2, 0.75), 
       legend.background = element_rect(fill = "white", color = "black"), 
        legend.title = element_text(face = "bold"),plot.title = element_text(hjust = 0.5))+
  ggtitle(expression("Zone 1")) +
  labs(x = "Time", y = "Incidence")

smr_pred2 <- cbind(as.data.frame(summary(fit_SEIRS, pars = "pred_Inc_cases2", probs = c(0.05, 0.5, 0.95))$summary), dates_pred[6:235])
colnames(smr_pred2) <- make.names(colnames(smr_pred2)) # to remove % in the col names
q<-ggplot(smr_pred2, mapping = aes(x = dates_pred[6:235])) +
  geom_ribbon(aes(ymin = X5., ymax = X95., fill = "95% CI"), alpha = 0.35) +
  geom_line(mapping = aes(x = dates_pred[6:235], y = X50., color = "brown4")) + 
  geom_line(mapping = aes(x = dates_pred[6:235], y = mean, color = "blue")) + 
  geom_point(data=yr1, mapping = aes(x=dates_pred[6:193], y =yr[,2], color = "black"), size=0.8) +
  geom_point(data=yr_2_zones_pred_data, mapping = aes(x=dates_pred[6:234], y =yr2_zones_pred_data[,2], color = "black"), size=0.8) +
  geom_vline(data=yr1, aes(xintercept=as.numeric(as.Date('2020-09-07')), colour="darkgreen"), linetype="solid", lwd=0.8, show.legend = F)+
  scale_x_date(date_breaks = "1 month", date_labels = "%Y-%m")+
  annotate(x = as.Date('2020-09-07'), y = +Inf, label = "2020-09-07", vjust = 2, geom = "label") +
  scale_color_identity(name = "Predictive check:",
                       breaks = c("gold2", "brown4", "blue", "black", "darkgreen"),
                       labels = c("95% CI", "Median model incidence", "Mean model incidence", "Observed daily incidence", "Prediction start date"),
                       guide = "legend")+
  #theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.position = c(0.20, 0.75), 
        legend.background = element_rect(fill = "white", color = "black"), 
        legend.title = element_text(face = "bold"),plot.title = element_text(hjust = 0.5))+
  ggtitle(expression("Zone 2")) +
  labs(x = "Time", y = "Incidence")

smr_pred3 <- cbind(as.data.frame(summary(fit_SEIRS, pars = "pred_Inc_cases3", probs = c(0.05, 0.5, 0.95))$summary), dates_pred[6:235])
colnames(smr_pred3) <- make.names(colnames(smr_pred3)) # to remove % in the col names
r<-ggplot(smr_pred3, mapping = aes(x = dates_pred[6:235])) +
  geom_ribbon(aes(ymin = X5., ymax = X95., fill = "95% CI"), alpha = 0.35) +
  geom_line(mapping = aes(x = dates_pred[6:235], y = X50., color = "brown4")) + 
  geom_line(mapping = aes(x = dates_pred[6:235], y = mean, color = "blue")) + 
  geom_point(data=yr1, mapping = aes(x=dates_pred[6:193], y = yr[,3], color = "black"), size=0.8) +
  geom_point(data=yr_2_zones_pred_data, mapping = aes(x=dates_pred[6:234], y = yr2_zones_pred_data[,3], color = "black"), size=0.8) +
  geom_vline(data=yr1, aes(xintercept=as.numeric(as.Date('2020-09-07')), colour="darkgreen"), linetype="solid", lwd=0.8, show.legend = F)+
  scale_x_date(date_breaks = "1 month", date_labels = "%Y-%m")+
  annotate(x = as.Date('2020-09-07'), y = +Inf, label = "2020-09-07", vjust = 2, geom = "label") +
  scale_color_identity(name = "Predictive check:",
                       breaks = c("gold2", "brown4", "blue", "black", "darkgreen"),
                       labels = c("95% CI", "Median model incidence", "Mean model incidence", "Observed daily incidence", "Prediction start date"),
                       guide = "legend")+
  #theme(plot.title = element_text(hjust = 0.5))+
 theme(legend.position = c(0.20, 0.75), 
        legend.background = element_rect(fill = "white", color = "black"), 
        legend.title = element_text(face = "bold"),plot.title = element_text(hjust = 0.5))+
  ggtitle(expression("Zone 3")) +
  labs(x ="Time", y = "Incidence")

smr_pred4 <- cbind(as.data.frame(summary(fit_SEIRS, pars = "pred_Inc_cases4", probs = c(0.05, 0.5, 0.95))$summary), dates_pred[6:235])
colnames(smr_pred4) <- make.names(colnames(smr_pred4)) # to remove % in the col names
s<-ggplot(smr_pred4, mapping = aes(x = dates_pred[6:235])) +
  geom_ribbon(aes(ymin = X5., ymax = X95., fill = "95% CI"), alpha = 0.35) +
  geom_line(mapping = aes(x = dates_pred[6:235], y = X50., color = "brown4")) + 
  geom_line(mapping = aes(x = dates_pred[6:235], y = mean, color = "blue")) + 
  geom_point(data=yr1, mapping = aes(x=dates_pred[6:193], y = yr[,4], color="black"), size=0.8) +
  geom_point(data=yr_2_zones_pred_data, mapping = aes(x=dates_pred[6:234], y = yr2_zones_pred_data[,4], color="black"), size=0.8) +
  geom_vline(data=yr1, aes(xintercept=as.numeric(as.Date('2020-09-07')), colour="darkgreen"), linetype="solid", lwd=0.8, show.legend = F)+
  scale_x_date(date_breaks = "1 month", date_labels = "%Y-%m")+
  annotate(x = as.Date('2020-09-07'), y = +Inf, label = "2020-09-07", vjust = 2, geom = "label") +
  scale_color_identity(name = "Predictive check:",
                       breaks = c("gold2", "brown4", "blue", "black", "darkgreen"),
                       labels = c("95% CI", "Median model incidence", "Mean model incidence", "Observed daily incidence", "Prediction start date"),
                       guide = "legend")+
  #theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.position = c(0.20, 0.75), 
       legend.background = element_rect(fill = "white", color = "black"), 
        legend.title = element_text(face = "bold"),plot.title = element_text(hjust = 0.5))+
  ggtitle(expression("Zone 4")) +
  labs(x = "Time", y = "Incidence")

#ggarrange(p,q,r,s, ncol = 2, nrow = 2)
ggarrange(p,q,r,s, ncol = 2, nrow = 2, common.legend = T)
ggsave("4zones_predictive_checks.png", width = 24, height = 18, units = c("cm"), dpi=300)


#Compute the efficiency measures

zone1_predicted_incidence_mean = smr_pred1$mean[194:229]
zone2_predicted_incidence_mean = smr_pred2$mean[194:229]
zone3_predicted_incidence_mean = smr_pred3$mean[194:229]
zone4_predicted_incidence_mean = smr_pred4$mean[194:229]

zone1_predicted_incidence_median = smr_pred1$X50.[194:229]
zone2_predicted_incidence_median = smr_pred2$X50.[194:229]
zone3_predicted_incidence_median = smr_pred3$X50.[194:229]
zone4_predicted_incidence_median = smr_pred4$X50.[194:229]

sum_all_zones_predicted_incidence_mean = zone1_predicted_incidence_mean + zone2_predicted_incidence_mean + zone3_predicted_incidence_mean + zone4_predicted_incidence_mean
sum_all_zones_predicted_incidence_median = zone1_predicted_incidence_median + zone2_predicted_incidence_median + zone3_predicted_incidence_median + zone4_predicted_incidence_median

observed_zone1_prediction_incidence = yr2_zones_pred_data[,1][194:229]
observed_zone2_prediction_incidence = yr2_zones_pred_data[,2][194:229]
observed_zone3_prediction_incidence = yr2_zones_pred_data[,3][194:229]
observed_zone4_prediction_incidence = yr2_zones_pred_data[,4][194:229]


#should be equal to global observed prediction incidence (not equal due to the smoothing sequence)
sum_all_zones_observed_prediction_incidence = observed_zone1_prediction_incidence + observed_zone2_prediction_incidence + observed_zone3_prediction_incidence + observed_zone4_prediction_incidence
observed_global_prediction_incidence #from single patch global incidence (run it from single_patch_SEIRS_Incidence_real_data.R)

n= length(sum_all_zones_predicted_incidence_mean)

sum_all_zones_squared_deviation_mean = (sum_all_zones_predicted_incidence_mean - observed_global_prediction_incidence)^2

sum_all_zones_sum_global_squared_deviation_mean = sum(sum_all_zones_squared_deviation_mean)

MSE_sum_all_zones_mean = sqrt(sum_all_zones_sum_global_squared_deviation_mean /n)
MSE_sum_all_zones_mean

sum_all_zones_relative_difference_mean = abs( (observed_global_prediction_incidence - sum_all_zones_predicted_incidence_mean)/observed_global_prediction_incidence   )
sum_sum_all_zones_relative_difference_mean = sum(sum_all_zones_relative_difference_mean)
MAPE_all_zones_mean = (sum_sum_all_zones_relative_difference_mean/n)*100
MAPE_all_zones_mean

sum_all_zones_squared_deviation_median = (sum_all_zones_predicted_incidence_median - observed_global_prediction_incidence)^2

sum_all_zones_sum_global_squared_deviation_median = sum(sum_all_zones_squared_deviation_median)

MSE_sum_all_zones_median = sqrt(sum_all_zones_sum_global_squared_deviation_median /n)
MSE_sum_all_zones_median

sum_all_zones_relative_difference_median = abs( (observed_global_prediction_incidence - sum_all_zones_predicted_incidence_median)/observed_global_prediction_incidence   )
sum_sum_all_zones_relative_difference_median = sum(sum_all_zones_relative_difference_median)
MAPE_all_zones_median = (sum_sum_all_zones_relative_difference_median/n)*100
MAPE_all_zones_median

#obtain the proportions covered by the 95% prediction intervals
zone1_lower_pred_incidence = smr_pred1$X5.[194:229]
zone1_upper_pred_incidence = smr_pred1$X95.[194:229]
zone2_lower_pred_incidence = smr_pred2$X5.[194:229]
zone2_upper_pred_incidence = smr_pred2$X95.[194:229]
zone3_lower_pred_incidence = smr_pred3$X5.[194:229]
zone3_upper_pred_incidence = smr_pred3$X95.[194:229]
zone4_lower_pred_incidence = smr_pred4$X5.[194:229]
zone4_upper_pred_incidence = smr_pred4$X95.[194:229]

count_observed_in_interval <- function(minvec, obsvec, maxvec) {
  
  n = length(obsvec) 
  # declaring sum =0 as the count of elements in range
  sum = 0
  
  # looping over the vector elements
  for(i in 1:n)
  {
    
    # check if elements lies in the range provided
    if(obsvec[i]>minvec[i] && obsvec[i]<maxvec[i])
      
      # incrementing count of sum if condition satisfied
      sum =sum+1
    
  }
  print("prop_observed_within_interval: ")
  print(sum/n)
  
}

zone1_prop_covered = count_observed_in_interval(zone1_lower_pred_incidence,observed_zone1_prediction_incidence,zone1_upper_pred_incidence )
zone2_prop_covered = count_observed_in_interval(zone2_lower_pred_incidence,observed_zone2_prediction_incidence,zone2_upper_pred_incidence )
zone3_prop_covered = count_observed_in_interval(zone3_lower_pred_incidence,observed_zone3_prediction_incidence,zone3_upper_pred_incidence )
zone4_prop_covered = count_observed_in_interval(zone4_lower_pred_incidence,observed_zone4_prediction_incidence,zone4_upper_pred_incidence )



