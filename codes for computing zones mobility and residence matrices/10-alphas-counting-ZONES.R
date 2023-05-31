# setwd("~/15-RTM-alphas/")

library(parallel)
library(magrittr)
library(dplyr)

PP <- matrix(c("FirstPeriod_FirstPart", "FF",
               "FirstPeriod_SecondPart", "FS",
               "SecondPeriod_FirstPart", "SF",
               "SecondPeriod_SecondPart", "SS",
               "ThirdPeriod_FirstPart", "TF",
               "ThirdPeriod_SecondPart", "TS"),
             ncol = 2, nrow = 6, 
             byrow = TRUE)

zones.file <- read.table("CVE_AGEB.csv", sep = ",", header = TRUE)

ageb.to.zone <- function(ageb, zones = zones.file){
  ageb %<>% as.integer()
  if(ageb == -1) return(-1)
  zones %>% 
    filter(ID_AGEB == ageb) %>% 
    select(ZONE) %>%
    as.numeric() %>%
    return()
}


for(p in 1:6){
  # Lectura de archivo CSV con en contenido de todos los id_adv que hicieron
  # ping en periodo y parte PP[p, ], asi como los agebs que visitaron 
  # (ageb_crit1, tercera columna) y los agebs en los que estuvieron en la
  # noche (ageb_crit2, segunda columna). En la columna loose (cuarta columna)
  # se se indica el ageb de residencia.

  PP.comb <- read.table(paste0("./final_time_periods/combined/", PP[p, 1], "_comb.csv"),
                        sep = ";",
                        header = TRUE) %>%
    select(X, loose)
  
  # Relacion de id_adv (y sus poligonos/agebs) identificados en PP
  PP.final <- read.table(paste0("./final_time_periods/criterion_1/", PP[p, 1], "_final.csv"),
                         sep = ";",
                         header = TRUE,
                         dec = ".") %>%
    select(id, polygon)
  

  no_cores <- detectCores()
  cl <- makeCluster(no_cores)
  clusterExport(cl, c("PP.comb", "PP.final", "zones.file", "ageb.to.zone", "%>%", "%<>%"))
  clusterEvalQ(cl, library(dplyr))
  
  PP.comb.zones <- parApply(cl,
                            PP.comb,
                            MARGIN = 1,
                            FUN = function(row)
                              return(c(row["X"], zone = ageb.to.zone(row["loose"])))) %>%
    t() %>%
    as.data.frame()
  
  PP.final.zones <- parApply(cl,
                             PP.final,
                             MARGIN = 1,
                             FUN = function(row)
                               return(c(row["id"], zone = ageb.to.zone(row["polygon"])))) %>%
    t() %>%
    as.data.frame()
  
  stopCluster(cl)
  
  
  # Lista de zonas
  zones <- c(-1, 1:4)
  
  no_cores <- detectCores()
  cl <- makeCluster(no_cores)
  clusterExport(cl, c("PP.comb.zones", "PP.final.zones", "zones", "%>%"))
  clusterEvalQ(cl, library(dplyr))
  
  # Numero de residentes de la i-esima zona que salen de esta:
  id_adv.fix <- rep(0, 5)
  
  pb <- txtProgressBar(min = 1, max = 5, style = 3)
  for(n in 1:5){
    # Para la i-esima zona, se identifican todos los id_adv que residen en ella:
    id_adv.n <- PP.comb.zones$X[PP.comb.zones$zone == zones[n]]
    
    # Para cada id_adv de los residentes (id_adv.n), se identifican las zonas unicas
    # por donde transito: PP.final.zones$zone[PP.final.zones$id == id_adv.n[*]]
    # Si el valor anterior es 1, se contabiliza en id_adv.fix[n]
    id_adv.fix[n] <- parLapply(cl, id_adv.n, function(id){
      count <- PP.final.zones$zone[PP.final.zones$id == id] %>%
        unique() %>%
        length()
      ifelse(count == 1, 1, 0) %>%
        return()
      }) %>%
      unlist() %>%
      sum()
    setTxtProgressBar(pb, n)
    }
  stopCluster(cl)
  
  # Numero de residentes en la i-esima zona; se inicializan en 0:
  id_adv.res <- rep(0, 5)
  
  # Numero de residentes por zona en PP:
  for(i in 1:5){
    id_adv.res[i] <- sum(PP.comb.zones$zone == zones[i])
  }
  
  write.csv(data.frame(zone = zones,
                       residents = id_adv.res,
                       permanecen_en_ageb = id_adv.fix, 
                       proporcion = id_adv.fix/id_adv.res), 
            file = paste0("./1-alphas-zones/alphas_", PP[p, 2], ".csv"), row.names = FALSE)
}
