##~~~~~~~~~~~~~ Importo librerias, funciones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rm(list=ls())

library(xgboost)
library(data.table)
library(tm)
library(Matrix)
library(dplyr)
library(ggplot2)
library(clue)
library(arules)
library(readxl)
library(stringr)
library(tm)

setwd("~/UTDT_MiM/UTDT - Mineria/")

#Importamos las funciones que tenemos en un script aparte
source("equipo_functions.R")
options(scipen=999)
##~~~~~~~~~~~~~ Cargamos los datos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_PATH <- "~/UTDT_MiM/UTDT - Mineria/competition_data/"

#Cargamos una muestra del 50% de los datos pero desde Mayo 2021
ads_data <- load_competition_data(DATA_PATH, sample_ratio = 0.3,from_when = "2021_05")
saveRDS(ads_data,"ads_data_75_2021_05.RDS")
ads_data <- readRDS("ads_data_75_2021_05.RDS")

#Eliminamos los datos de fin de septiembre (en verdad no son 0)
ads_data <- ads_data[!((strftime(ads_data$created_on, "%Y-%m", tz="UTC") == "2021-09") & (strftime(ads_data$created_on, "%d", tz="UTC") >= 17)),]

##~~~~~~~~~~~~~ Guardamos una variable que identifique training, validation y testing ~~~~~~~~~~~~~
ads_data$train_val_eval <- ifelse(ads_data$created_on >= strptime("2021-10-01", format = "%Y-%m-%d", tz = "UTC"), "eval", "train")
ads_data[sample(which(ads_data$train_val_eval == "train"), round(0.05 * sum(ads_data$train_val_eval == "train"))), "train_val_eval"] <- "valid"

##~~~~~~~~~~~~~ Feature engineering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Creamos variables a partir de la fecha para poder captar comportamientos estilo estacionalidades
ads_data$day <- as.integer(strftime(ads_data$created_on, format = "%d", tz = "UTC")) #Dia
ads_data$month <- as.integer(strftime(ads_data$created_on, format = "%m", tz = "UTC")) #Mes
ads_data$year <- as.integer(strftime(ads_data$created_on, format = "%Y", tz = "UTC")) #Anio
ads_data$week_day <- as.integer(strftime(ads_data$created_on, format = "%w", tz = "UTC")) #Dia de la semana
ads_data$year_week <- as.integer(strftime(ads_data$created_on, format = "%W", tz = "UTC")) #Semana del anio
ads_data$fin_de_semana <- (ifelse(ads_data$week_day %in% c(1,2,3,4,5), 0, 1)) #agregado

#Analizamos la variabilidad de las variables
mean(is.na(ads_data$place_l6)) #place_l6 tiene 97% de NA
mean(is.na(ads_data$place_l5)) #place_l5 tiene 87% de NA
mean(is.na(ads_data$place_l4)) #place_l5 tiene 65% de NA
mean(is.na(ads_data$current_state)) # current_state tiene 99% de NA
mean(is.na(ads_data$surface_total)) # current_state tiene 77% de NA
mean(is.na(ads_data$surface_covered)) # current_state tiene 75% de NA

unique(ads_data$place_l6)
unique(ads_data$place_l5)
unique(ads_data$place_l4)
unique(ads_data$current_state)

ads_data$place_l6 <- ifelse(is.na(ads_data$place_l6), 0, 1)
ads_data$place_l5 <- ifelse(is.na(ads_data$place_l5), 0, 1)
ads_data$place_l4 <- ifelse(is.na(ads_data$place_l4), 0, 1) #agregamos place l4

ads_data$sin_terminar <- ifelse(ads_data$current_state %in% c('En Planos', 'En construcción','En Obra', 'En Pozo',
                                                              'Construyendo'), 1, 0) #agregamos si falta que termine

#bedrooms negativos corrijo
ads_data <- ads_data %>% mutate(bedrooms= ifelse(bedrooms == -3, 3, bedrooms))
ads_data <- ads_data %>% mutate(bedrooms= ifelse(bedrooms == -2, 2, bedrooms))
ads_data <- ads_data %>% mutate(bedrooms= ifelse(bedrooms == -1, 1, bedrooms))

#surface_covered negativos corrijo
ads_data <- ads_data %>% mutate(surface_covered= ifelse(surface_covered == -3, 3, surface_covered))
ads_data <- ads_data %>% mutate(surface_covered= ifelse(surface_covered == -2, 2, surface_covered))

#bedrooms = 0

table(ads_data$property_type,is.na(ads_data$bedrooms))
ads_data <- ads_data %>% mutate(bedrooms= ifelse(property_type %in% c('Local comercial','Oficina','Lote', 'Cochera', 'Depósito') & is.na(bedrooms),0,bedrooms))
table(ads_data$property_type,is.na(ads_data$bedrooms)) #reviso nulos ok

#bathrooms = 0

table(ads_data$property_type,is.na(ads_data$bathrooms))
ads_data <- ads_data %>% mutate(bathrooms= ifelse(property_type %in% c('Lote', 'Cochera', 'Depósito') & is.na(bathrooms),0,bathrooms))
table(ads_data$property_type,is.na(ads_data$bathrooms))

#variable dicotomica tiene/no tiene bath y bed
ads_data$has_bath <- ifelse(ads_data$property_type %in% c('Departamento','Local comercial','Oficina','Depósito','Casa','PH','Casa de campo'), 1,0)
ads_data$has_bed <- ifelse(ads_data$property_type %in% c('Departamento','Casa','PH','Casa de campo'), 1,0)

table(ads_data$property_type,ads_data$has_bath) #ojo con otros

#Podemos aprovechar el texto!
ads_data$pileta <- ifelse(grepl(paste(c("pileta", "piscina"), collapse = "|"), tolower(ads_data$description)) == TRUE, 1,0)
ads_data$nuevo <- ifelse(grepl(paste(c("nuev", "estren"), collapse = "|"), tolower(ads_data$description)) == TRUE, 1,0) #agregue
ads_data$jardin <- ifelse(grepl("jardin", chartr("ÁÉÍÓÚ", "AEIOU", tolower(ads_data$description))) == TRUE, 1,0)
ads_data$mascota <- ifelse(grepl(paste(c("mascota", "perro", "gato"), collapse = "|"), tolower(ads_data$description)) == TRUE, 1,0) #agregue

##~~~~~~~~~~~~~ Bag of Words ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bow_title <- gen_bow_vars(ads_data$title, "title", 4, 3000, FALSE)
head(bow_title, 300)
saveRDS(bow_title,"bow_title_75_2021_05.RDS")
bow_title <- readRDS("bow_title_75_2021_05.RDS")

#Sacamos la columna de ttitle y description que es demasiado verbosa y no agregaría al análisis
ads_data <- subset(ads_data, select = -c(description))
saveRDS(ads_data,"ads_data_75_2021_05.RDS")
##~~~~~~~~~~~~~ Ratios ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Agregamos combinaciones de variables y ratios
ads_data <- ads_data %>% mutate(bed_and_bath = bathrooms + bedrooms,
                                rooms_bed = round(rooms/bedrooms,2),
                                rooms_bath = round(rooms/bathrooms,2),
                                bed_bath = round(bedrooms/bathrooms,2),
                                surfacecovered_surfacetotal = round(surface_covered/surface_total,2),
                                surfacecovered_rooms = round(surface_covered/rooms,2),
                                surfaceuncovered =  surface_total-surface_covered,
                                price_m2_covered = round(price_usd/surface_covered,2),
                                price_m2 = round(price_usd/surface_total,2),
                                price_m2_uncovered = round(price_usd/surfaceuncovered,2),
                                price_bath = round(price_usd/bathrooms,2),
                                price_bedrooms = round(price_usd/bedrooms,2),
                                price_bed_and_bath = round(price_usd/bed_and_bath,2),
                                price_rooms = round(price_usd/rooms,2))

ratios = c("rooms_bed", "rooms_bath", "bed_bath", "surfacecovered_surfacetotal", "surfacecovered_rooms",
           "surfaceuncovered","price_m2_covered", "price_m2","price_m2_uncovered","price_bath",
           "price_bedrooms","price_bed_and_bath","price_rooms")

#Algunos denomidadores son 0, corrijo Inf
ads_data[, ratios] <- lapply(ads_data[, ..ratios], function(x) ifelse(is.infinite(x), NA, x))

##~~~~~~~~~~~~~ Fuentes externas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Utilizamos indice de movilidad de Google como fuente externa de información
mobility <- readRDS("mobility.RDS")
mobility$country_region <- as.character(mobility$country_region)
mobility["country_region"][mobility["country_region"]=='Peru'] <- 'Perú'
mobility$country_region <- as.factor(mobility$country_region)
mobility$date <- strftime(mobility$date, "%Y-%m-%d", tz="UTC")
mobility <- as.data.table(mobility)
ads_data$date <- strftime(ads_data$created_on, "%Y-%m-%y", tz="UTC")

#Mergeamos score de mobility con ads_data
ads_data <- merge(ads_data,mobility, by.x = c("place_l1", "date"), by.y = c("country_region", "date"), all.x = TRUE)

#Eliminamos mobility
rm(mobility)

##~~~~~~~~~~~~~ Aprendizaje no supervisado ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Probemos crear variables utilizando aprendizaje no supervisado

#La fecha como numerica
ads_data$created_on <- as.numeric(ads_data$created_on)

train_data <- ads_data[ads_data$train_val_eval == "train"]
kmeans_columns_to_keep <- c('price_usd', 'price_bedrooms', 'price_bed_and_bath', 'lon', 'lat', 'price', 'price_bath',
                            'created_on') #uso los primeros del grafico
train_data_clusters <- train_data[, ..kmeans_columns_to_keep]
train_data_clusters <- train_data_clusters[complete.cases(train_data_clusters)]

#Escalamos y eliminamos outliers
train_data_clusters <- as.data.frame(sapply(train_data_clusters, function(data) (abs(data-mean(data))/sd(data))))    
train_data_clusters <- train_data_clusters[!rowSums(train_data_clusters>10),]

kmeans_results <- find_k_means(train_data_clusters,3,20)
saveRDS(kmeans_results,"kmeans_result.RDS")
kmeans_results <- readRDS("kmeans_result.RDS")

plot(c(1:18), kmeans_results$var, type="o",
     xlab="# Clusters", ylab="tot.withinss")

#Entrenamos KMeans
clusters_model <- kmeans(train_data_clusters,
                         centers= 4, iter.max=3000, nstart=10)

#Hay diferencias entre grupos?
train_data$cluster <- factor(cl_predict(clusters_model, train_data[, ..kmeans_columns_to_keep]))
saveRDS(train_data,"train_data_clusters.RDS")
train_data <- readRDS("train_data_clusters.RDS")

clusterizacion <- train_data %>% group_by(cluster) %>% summarise(prom_contactos = mean(contacts, na.rm=TRUE),
                                  cant_obs = n(),
                                  cant_obs_pct = n()/nrow(.))

#Predecimos el cluster para todo el dataframe
ads_data$cluster <- factor(cl_predict(clusters_model, ads_data[, ..kmeans_columns_to_keep]))

table(is.na(ads_data$cluster))

##~~~~~~~~~~~~~ Hacemos one_hot_encoding y pasamos a matrices ralas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train_val_eval <- ads_data$train_val_eval
cols_to_delete <- c("ad_id","title", "description", "short_description", "development_name","train_val_eval") #ad_id probar sacarlo tambien
columns_to_keep <- setdiff(names(ads_data), cols_to_delete)
ads_data_sparse <- one_hot_sparse(ads_data[, ..columns_to_keep])
gc()
saveRDS(ads_data_sparse,"ads_data_sparse_combinado.RDS")
ads_data_sparse <- readRDS("ads_data_sparse_combinado.RDS")
ads_data %>% group_by((cluster)) %>% summarise(prom_contactos = mean(contacts, na.rm=TRUE),
                                               cant_obs = n(),
                                               cant_obs_pct = n()/nrow(.))
saveRDS(ads_data,"ads_data_pretraining.RDS")
rm(ads_data)
##~~~~~~~~~~~~~ Combinamos matrices ralas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ads_data_sparse <- cbind(ads_data_sparse,bow_title)

##~~~~~~~~~~~~~ Entrenamos modelo de xgboost ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dtrain <- xgb.DMatrix(data = ads_data_sparse[train_val_eval == "train", colnames(ads_data_sparse) != "contacts"],
                      label = ads_data_sparse[train_val_eval == "train", colnames(ads_data_sparse) == "contacts"])

dvalid <- xgb.DMatrix(data = ads_data_sparse[train_val_eval == "valid", colnames(ads_data_sparse) != "contacts"],
                      label = ads_data_sparse[train_val_eval == "valid", colnames(ads_data_sparse) == "contacts"])

rgrid <- random_grid(size = 10, #cant resultados finales --> entre 10 y 20
                     min_nrounds = 150, max_nrounds = 450, #cantidad de arboles (0,Inf) #aprox 2 horas
                     min_max_depth = 8, max_max_depth = 20, #profundidad de los arboles (0,Inf) #probar con valor superior a 15
                     min_eta = 0.01, max_eta = 0.5, #learning rate (0,1]
                     min_gamma = 0, max_gamma = 1, #regularizador de complejidad del modelo (0,Inf) #Aumentar
                     min_min_child_weight = 2, max_min_child_weight = 8, #numero minimo de obs en una hoja para crear hijo (0,Inf)
                     min_colsample_bytree = 0.2, max_colsample_bytree = 0.6, #columnas sampleadas por arbol (0,1]
                     min_subsample = 0.3, max_subsample = 0.8) #observaciones sampleadas por arbol (0,1]

predicted_models <- train_xgboost(dtrain, dvalid, rgrid)

saveRDS(predicted_models, "predicted_models_combinado.RDS")
predicted_models <- readRDS("predicted_models_combinado.RDS")

# Guardamos los resultados en un dataframe de una forma comoda de verlo
res_table <- result_table(predicted_models)
print(res_table)

#Nos quedamos con el mejor modelo
best_model <- predicted_models[[res_table[1,"i"]]]$model

#Analizamos las variables con mayor poder predictivo
importance_matrix = xgb.importance(colnames(dtrain), model = best_model)
xgb.plot.importance(importance_matrix[1:30,])

##~~~~~~~~~~~~~ Reentrenamos con train + validation para predecir eval ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dall <- xgb.DMatrix(data = ads_data_sparse[train_val_eval != "eval", colnames(ads_data_sparse) != "contacts"],
                    label = ads_data_sparse[train_val_eval != "eval", colnames(ads_data_sparse) == "contacts"])
final_model <- xgb.train(data = dall,
                         nrounds = res_table[1, "nrounds"],
                         params=as.list(res_table[1, c("max_depth",
                                                       "eta",
                                                       "gamma",
                                                       "colsample_bytree",
                                                       "subsample",
                                                       "min_child_weight")]),
                         watchlist = list(train = dall),
                         objective = "reg:squaredlogerror",
                         feval = rmsle,
                         print_every_n = 10)

##~~~~~~~~~~~~~ Resultados ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##Predecimos en eval y guardamos submissions
ads_data <- readRDS("ads_data_pretraining.RDS")
preds <- data.frame(ad_id = ads_data[train_val_eval == "eval", "ad_id"],
                    contacts = predict(final_model,
                                       ads_data_sparse[train_val_eval == "eval", colnames(ads_data_sparse) != "contacts"]))
preds$contacts <- pmax(preds$contacts, 0)

options(scipen=10)
write.table(preds, "submission_combinado.csv", sep=",", row.names=FALSE, quote=FALSE)