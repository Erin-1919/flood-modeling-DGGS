library (VSURF)
library (ROCR)
library (dplyr)

## Data prep
# read sample points
spoints.21 = read.csv("../Data/NB_sample_cent_quanti3_21.csv",header=TRUE,fileEncoding="UTF-8-BOM")
spoints.21 = subset(spoints.21, select=-c(i,j,Cell_address,lon_c,lat_c))
spoints.21 = spoints.21 %>% filter_all(all_vars(!grepl(-32767, .)))
spoints.21 = spoints.21[complete.cases(spoints.21), ]

# convert to factor
cols = c("grid_code", "geo", "wl", "sol", "lc", "ia", "fcp")
spoints.21[cols] = lapply(spoints.21[cols], factor)
sapply(spoints.21, class)

# reorder
col_order = c("grid_code","geo","wl","sol","lc","ndvi","msi","ia","fcp","dtm","slp","asp","curv","rgh","tri","tpi","nhn",
              "precip","tavg","r10","r25","tm10","sd50","ts","spr")
var_order = c("geo","wl","sol","lc","ndvi","msi","ia","fcp","dtm","slp","asp","curv","rgh","tri","tpi","nhn",
              "precip","tavg","r10","r25","tm10","sd50","ts","spr")
  
spoints.21 = spoints.21[, col_order]

# randomly select 70 percent training data, left 30 percent for testing data
spoints.21.train = spoints.21 %>% sample_frac(.7)
spoints.21.test = setdiff(spoints.21,spoints.21.train)

## RF model
set.seed(500)

# 20 hydro-geomorphological variables
vnb.HG.21 = VSURF(x = spoints.21.train[,2:17], y = spoints.21.train[,1], nmin = 3, ntree = 500, 
                  parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action = na.omit)

summary(vnb.HG.21)
print(vnb.HG.21$mean.perf)
plot(vnb.HG.21,var.names=TRUE)

# 20 hydro-geomorphological variables + average annual Precipitation and Temperature
vnb.HGPT.21 = VSURF(x = spoints.21.train[,2:19], y = spoints.21.train[,1], nmin = 3, ntree = 500, 
                    parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action=na.omit)

summary(vnb.HGPT.21)
print(vnb.HGPT.21$mean.perf)
plot(vnb.HGPT.21,var.names=TRUE)

# all variables
vnb.HG8M.21 = VSURF(x = spoints.21.train[,2:25], y = spoints.21.train[,1], nmin = 3, ntree = 500, 
                    parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action=na.omit)

summary(vnb.HG8M.21)
print(vnb.HG8M.21$mean.perf)
plot(vnb.HG8M.21,var.names=TRUE)

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

print(index_to_var(vnb.HG.21$varselect.interp))
print(index_to_var(vnb.HG.21$varselect.pred))
print(index_to_var(vnb.HGPT.21$varselect.interp))
print(index_to_var(vnb.HGPT.21$varselect.pred))
print(index_to_var(vnb.HG8M.21$varselect.interp))
print(index_to_var(vnb.HG8M.21$varselect.pred))

## Evaluating and visualizing the performance
# HG
vnb.HG.21.pred = predict(vnb.HG.21, newdata = spoints.21.test[,2:17], step = c("interp","pred"))
pred.21 = ROCR::prediction(as.numeric(vnb.HG.21.pred$pred),as.numeric(spoints.21.test$grid_code))
performance(pred.21,"acc")@y.values[[1]][2]
performance(pred.21,"auc")@y.values[[1]]
performance(pred.21,"f")@y.values[[1]][2]

# HGPT
vnb.HGPT.21.pred = predict(vnb.HGPT.21, newdata = spoints.21.test[,2:19], step = c("interp","pred"))
pred.21 = ROCR::prediction(as.numeric(vnb.HGPT.21.pred$pred),as.numeric(spoints.21.test$grid_code))
performance(pred.21,"acc")@y.values[[1]][2]
performance(pred.21,"auc")@y.values[[1]]
performance(pred.21,"f")@y.values[[1]][2]

# HG8M
vnb.HG8M.21.pred = predict(vnb.HG8M.21, newdata = spoints.21.test[,2:25], step = c("interp","pred"))
pred.21 = ROCR::prediction(as.numeric(vnb.HG8M.21.pred$pred),as.numeric(spoints.21.test$grid_code))
performance(pred.21,"acc")@y.values[[1]][2]
performance(pred.21,"auc")@y.values[[1]]
performance(pred.21,"f")@y.values[[1]][2]

## Prediction over full study area
# read all centroid points
cent.21 = read.csv("../Data/NB_finalPredict_21.csv",header=TRUE,fileEncoding="UTF-8-BOM")
cent.21 = subset(cent.21, select=-c(i,j,Cell_address))
cent.21 = cent.21 %>% filter_all(all_vars(!grepl(-32767, .)))
cent.21 = cent.21[complete.cases(cent.21), ]

# convert to factor
cols = c("geo", "wl", "sol", "lc", "ia", "fcp")
cent.21[cols] = lapply(cent.21[cols], factor)
cent.21$HG = predict(vnb.HG.21, newdata = cent.21[,3:26], step = c("pred"))
cent.21$HGPT = predict(vnb.HGPT.21, newdata = cent.21[,3:26], step = c("pred"))
cent.21$HG8M = predict(vnb.HG8M.21, newdata = cent.21[,3:26], step = c("pred"))

pred.result.21 = select(cent.21,c('lon_c','lat_c','HG','HGPT','HG8M'))
write.csv(pred.result.21,"../Result/NB_centPredict_21.csv", row.names = FALSE)
