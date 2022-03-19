library (VSURF)
library (ROCR)
library (dplyr)

# res_ls = c(19,21,23)
dggs_res = 19
sub_id_ls = c(1,2,3,4,5,8,9,10,11,13,14,16)
col_order = c("grid_code","geo","wl","sol","lc","ndvi","msi","ia","fcp","dtm","slp","asp","curv","rgh","tri","tpi","nhn","fldir","flacc","twi","spi",
              "precip","tavg","r10","r25","tm10","sd50","ts","spr")
var_order = c("geo","wl","sol","lc","ndvi","msi","ia","fcp","dtm","slp","asp","curv","rgh","tri","tpi","nhn","fldir","flacc","twi","spi",
              "precip","tavg","r10","r25","tm10","sd50","ts","spr")

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


CombPoints = setNames(data.frame(matrix(ncol = 29, nrow = 0)), col_order)

for (sub_id in sub_id_ls) {
  spoints = read.csv(sprintf("../Data/sub_basin/NB_sample_cent_quanti4_%d_%d.csv",sub_id,dggs_res),header=TRUE,fileEncoding="UTF-8-BOM")
  spoints = subset(spoints, select=-c(i,j,Cell_address,lon_c,lat_c))
  spoints = filter(spoints, geo != -32767 & dtm != -32767)
  spoints = spoints[complete.cases(spoints), ]
  
  # convert to factor
  cols = c("grid_code", "geo", "wl", "sol", "lc", "ia", "fcp")
  spoints[cols] = lapply(spoints[cols], factor)
  sapply(spoints, class)
  
  # reorder
  spoints = spoints[, col_order]
  
  # combine dataframes
  CombPoints = rbind(CombPoints,spoints)
}

rm(spoints)

# randomly select 70 percent training data, left 30 percent for testing data
CombPoints.train = CombPoints %>% sample_frac(.7)
CombPoints.test = setdiff(CombPoints,CombPoints.train)

## RF model
set.seed(500)

# 20 hydro-geomorphological variables
vnb.HG = VSURF(x = CombPoints.train[,2:21], y = CombPoints.train[,1], nmin = 3, ntree = 500, 
               parallel = TRUE, ncores = 4, clusterType = "PSOCK", na.action = na.omit)

print(index_to_var(vnb.HG$varselect.interp))
print(index_to_var(vnb.HG$varselect.pred))

# 20 hydro-geomorphological variables + average annual Precipitation and Temperature
vnb.HGPT = VSURF(x = CombPoints.train[,2:23], y = CombPoints.train[,1], nmin = 3, ntree = 500, 
                 parallel = TRUE, ncores = 4, clusterType = "PSOCK", na.action=na.omit)

print(index_to_var(vnb.HGPT$varselect.interp))
print(index_to_var(vnb.HGPT$varselect.pred))

# all variables
vnb.HG8M = VSURF(x = CombPoints.train[,2:29], y = CombPoints.train[,1], nmin = 3, ntree = 500, 
                 parallel = TRUE, ncores = 4, clusterType = "PSOCK", na.action=na.omit)

print(index_to_var(vnb.HG8M$varselect.interp))
print(index_to_var(vnb.HG8M$varselect.pred))


## Evaluating performance
# HG
vnb.HG.pred = predict(vnb.HG, newdata = CombPoints.test[,2:21], step = c("interp","pred"))
pred = ROCR::prediction(as.numeric(vnb.HG.pred$pred),as.numeric(CombPoints.test$grid_code))
performance(pred,"acc")@y.values[[1]][2]
performance(pred,"auc")@y.values[[1]]
performance(pred,"f")@y.values[[1]][2]

# HGPT
vnb.HGPT.pred = predict(vnb.HGPT, newdata = CombPoints.test[,2:23], step = c("interp","pred"))
pred = ROCR::prediction(as.numeric(vnb.HGPT.pred$pred),as.numeric(CombPoints.test$grid_code))
performance(pred,"acc")@y.values[[1]][2]
performance(pred,"auc")@y.values[[1]]
performance(pred,"f")@y.values[[1]][2]

# HG8M
vnb.HG8M.pred = predict(vnb.HG8M, newdata = CombPoints.test[,2:29], step = c("interp","pred"))
pred = ROCR::prediction(as.numeric(vnb.HG8M.pred$pred),as.numeric(CombPoints.test$grid_code))
performance(pred,"acc")@y.values[[1]][2]
performance(pred,"auc")@y.values[[1]]
performance(pred,"f")@y.values[[1]][2]
