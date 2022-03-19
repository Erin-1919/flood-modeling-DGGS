library (VSURF)
library (ROCR)
library (dplyr)

## Data prep
# read sample points
spoints.23 = read.csv("../Data/NB_sample_cent_quanti3_23.csv",header=TRUE,fileEncoding="UTF-8-BOM")
spoints.23 = subset(spoints.23, select=-c(i,j,Cell_address,lon_c,lat_c))
spoints.23 = spoints.23 %>% filter_all(all_vars(!grepl(-32767, .)))
spoints.23 = spoints.23[complete.cases(spoints.23), ]

# convert to factor
cols = c("grid_code", "geo", "wl", "sol", "lc", "ia", "fcp")
spoints.23[cols] = lapply(spoints.23[cols], factor)
sapply(spoints.23, class)

# reorder
col_order = c("grid_code","geo","wl","sol","lc","ndvi","msi","ia","fcp","dtm","slp","asp","curv","rgh","tri","tpi","nhn",
              "precip","tavg","r10","r25","tm10","sd50","ts","spr")
var_order = c("geo","wl","sol","lc","ndvi","msi","ia","fcp","dtm","slp","asp","curv","rgh","tri","tpi","nhn",
              "precip","tavg","r10","r25","tm10","sd50","ts","spr")

spoints.23 = spoints.23[, col_order]

# randomly select 70 percent training data, left 30 percent for testing data
spoints.23.train = spoints.23 %>% sample_frac(.7)
spoints.23.test = setdiff(spoints.23,spoints.23.train)

## RF model
set.seed(500)

# 20 hydro-geomorphological variables
vnb.HG.23 = VSURF(x = spoints.23.train[,2:17], y = spoints.23.train[,1], nmin = 3, ntree = 500, 
                  parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action = na.omit)

summary(vnb.HG.23)
print(vnb.HG.23$mean.perf)
plot(vnb.HG.23,var.names=TRUE)

# 20 hydro-geomorphological variables + average annual Precipitation and Temperature
vnb.HGPT.23 = VSURF(x = spoints.23.train[,2:19], y = spoints.23.train[,1], nmin = 3, ntree = 500, 
                    parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action=na.omit)

summary(vnb.HGPT.23)
print(vnb.HGPT.23$mean.perf)
plot(vnb.HGPT.23,var.names=TRUE)

# all variables
vnb.HG8M.23 = VSURF(x = spoints.23.train[,2:25], y = spoints.23.train[,1], nmin = 3, ntree = 500, 
                    parallel = TRUE, ncores = 8, clusterType = "PSOCK", na.action=na.omit)

summary(vnb.HG8M.23)
print(vnb.HG8M.23$mean.perf)
plot(vnb.HG8M.23,var.names=TRUE)

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

print(index_to_var(vnb.HG.23$varselect.interp))
print(index_to_var(vnb.HG.23$varselect.pred))
print(index_to_var(vnb.HGPT.23$varselect.interp))
print(index_to_var(vnb.HGPT.23$varselect.pred))
print(index_to_var(vnb.HG8M.23$varselect.interp))
print(index_to_var(vnb.HG8M.23$varselect.pred))

## Evaluating and visualizing the performance
# HG
vnb.HG.23.pred = predict(vnb.HG.23, newdata = spoints.23.test[,2:17], step = c("interp","pred"))
pred.23 = ROCR::prediction(as.numeric(vnb.HG.23.pred$pred),as.numeric(spoints.23.test$grid_code))
performance(pred.23,"acc")@y.values[[1]][2]
performance(pred.23,"auc")@y.values[[1]]
performance(pred.23,"f")@y.values[[1]][2]

# HGPT
vnb.HGPT.23.pred = predict(vnb.HGPT.23, newdata = spoints.23.test[,2:19], step = c("interp","pred"))
pred.23 = ROCR::prediction(as.numeric(vnb.HGPT.23.pred$pred),as.numeric(spoints.23.test$grid_code))
performance(pred.23,"acc")@y.values[[1]][2]
performance(pred.23,"auc")@y.values[[1]]
performance(pred.23,"f")@y.values[[1]][2]

# HG8M
vnb.HG8M.23.pred = predict(vnb.HG8M.23, newdata = spoints.23.test[,2:25], step = c("interp","pred"))
pred.23 = ROCR::prediction(as.numeric(vnb.HG8M.23.pred$pred),as.numeric(spoints.23.test$grid_code))
performance(pred.23,"acc")@y.values[[1]][2]
performance(pred.23,"auc")@y.values[[1]]
performance(pred.23,"f")@y.values[[1]][2]

## Prediction over full study area
# read all centroid points
cent.23 = read.csv("../Data/NB_finalPredict_23.csv",header=TRUE,fileEncoding="UTF-8-BOM")
cent.23 = subset(cent.23, select=-c(i,j,Cell_address))
cent.23 = cent.23 %>% filter_all(all_vars(!grepl(-32767, .)))
cent.23 = cent.23[complete.cases(cent.23), ]

# convert to factor
cols = c("geo", "wl", "sol", "lc", "ia", "fcp")
cent.23[cols] = lapply(cent.23[cols], factor)
cent.23$HG = predict(vnb.HG.23, newdata = cent.23[,3:26], step = c("pred"))
cent.23$HGPT = predict(vnb.HGPT.23, newdata = cent.23[,3:26], step = c("pred"))
cent.23$HG8M = predict(vnb.HG8M.23, newdata = cent.23[,3:26], step = c("pred"))

pred.result.23 = select(cent.23,c('lon_c','lat_c','HG','HGPT','HG8M'))
write.csv(pred.result.23,"../Result/NB_centPredict_23.csv", row.names = FALSE)
save.image(file = "test_nb_vsurf_pred_23.RData")