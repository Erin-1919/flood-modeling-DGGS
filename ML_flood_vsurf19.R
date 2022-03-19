library (VSURF)
library (ROCR)
library (dplyr)

## Data prep
# read sample points
spoints.19 = read.csv("../Data/NB_sample_cent_quanti4_19.csv",header=TRUE,fileEncoding="UTF-8-BOM")
spoints.19 = subset(spoints.19, select=-c(i,j,Cell_address,lon_c,lat_c))
spoints.19 = filter(spoints.19, geo != -32767 & dtm != -32767)
spoints.19 = spoints.19[complete.cases(spoints.19), ]

# convert to factor
cols = c("grid_code", "geo", "wl", "sol", "lc", "ia", "fcp")
spoints.19[cols] = lapply(spoints.19[cols], factor)
sapply(spoints.19, class)

# reorder
col_order = c("grid_code","geo","wl","sol","lc","ndvi","msi","ia","fcp","dtm","slp","asp","curv","rgh","tri","tpi","nhn","fldir","flacc","twi","spi",
              "precip","tavg","r10","r25","tm10","sd50","ts","spr")
var_order = c("geo","wl","sol","lc","ndvi","msi","ia","fcp","dtm","slp","asp","curv","rgh","tri","tpi","nhn","fldir","flacc","twi","spi",
              "precip","tavg","r10","r25","tm10","sd50","ts","spr")

spoints.19 = spoints.19[, col_order]

# randomly select 70 percent training data, left 30 percent for testing data
spoints.19.train = spoints.19 %>% sample_frac(.7)
spoints.19.test = setdiff(spoints.19,spoints.19.train)

## RF model
set.seed(500)

# 20 hydro-geomorphological variables
vnb.HG.19 = VSURF(x = spoints.19.train[,2:21], y = spoints.19.train[,1], nmin = 3, ntree = 500, 
                  parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action = na.omit)

summary(vnb.HG.19)
print(vnb.HG.19$mean.perf)
plot(vnb.HG.19,var.names=TRUE)

# 20 hydro-geomorphological variables + average annual Precipitation and Temperature
vnb.HGPT.19 = VSURF(x = spoints.19.train[,2:23], y = spoints.19.train[,1], nmin = 3, ntree = 500, 
                    parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action=na.omit)

summary(vnb.HGPT.19)
print(vnb.HGPT.19$mean.perf)
plot(vnb.HGPT.19,var.names=TRUE)

# all variables
vnb.HG8M.19 = VSURF(x = spoints.19.train[,2:29], y = spoints.19.train[,1], nmin = 3, ntree = 500, 
                    parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action=na.omit)

summary(vnb.HG8M.19)
print(vnb.HG8M.19$mean.perf)
plot(vnb.HG8M.19,var.names=TRUE)

## create a function to map index to variable name
index_to_var <- function(index_vec) {
  var_name_vec = c()
  for (i in index_vec)
  {
    var_name = var_order[i]
    var_name_vec = append(var_name_vec,var_name)
  }
  return(var_name_vec)
}

print(index_to_var(vnb.HG.19$varselect.interp))
print(index_to_var(vnb.HG.19$varselect.pred))
print(index_to_var(vnb.HGPT.19$varselect.interp))
print(index_to_var(vnb.HGPT.19$varselect.pred))
print(index_to_var(vnb.HG8M.19$varselect.interp))
print(index_to_var(vnb.HG8M.19$varselect.pred))

## Evaluating and visualizing the performance
# HG
vnb.HG.19.pred = predict(vnb.HG.19, newdata = spoints.19.test[,2:21], step = c("interp","pred"))
pred.19 = ROCR::prediction(as.numeric(vnb.HG.19.pred$pred),as.numeric(spoints.19.test$grid_code))
performance(pred.19,"acc")@y.values[[1]][2]
performance(pred.19,"auc")@y.values[[1]]
performance(pred.19,"f")@y.values[[1]][2]

# HGPT
vnb.HGPT.19.pred = predict(vnb.HGPT.19, newdata = spoints.19.test[,2:23], step = c("interp","pred"))
pred.19 = ROCR::prediction(as.numeric(vnb.HGPT.19.pred$pred),as.numeric(spoints.19.test$grid_code))
performance(pred.19,"acc")@y.values[[1]][2]
performance(pred.19,"auc")@y.values[[1]]
performance(pred.19,"f")@y.values[[1]][2]

# HG8M
vnb.HG8M.19.pred = predict(vnb.HG8M.19, newdata = spoints.19.test[,2:29], step = c("interp","pred"))
pred.19 = ROCR::prediction(as.numeric(vnb.HG8M.19.pred$pred),as.numeric(spoints.19.test$grid_code))
performance(pred.19,"acc")@y.values[[1]][2]
performance(pred.19,"auc")@y.values[[1]]
performance(pred.19,"f")@y.values[[1]][2]