setwd("E:/master/ray_results/")


normalize <- function(x) {
  return((x- min(x)) /(max(x)-min(x)))
}


# Calculate maximums 
version <- "buffer"
subs <- c("unet", "denseunet", "aspp", "attention", "stacked")


t_dicesb = c()
t_mae = c()
t_medae = c()
t_rmse = c()
t_ssim = c()
t_zncc = c()

v_dicesb = c()
v_mae = c()
v_medae = c()
v_rmse = c()
v_ssim = c()
v_zncc = c()

f_dicesb = c()
f_mae = c()
f_medae = c()
f_rmse = c()
f_ssim = c()
f_zncc = c()
f_comp = c()
f_corr = c()
f_qual = c()


for(sub in subs){
  dir <- paste0("runs_5e-05/seed/final/", sub, "/", version, "/progress.csv")
  
  data <- read.csv(dir)
  
  index <- which.min(data$validation_loss)
  
  t_dicesb <- c(data$training_loss[index], t_dicesb)
  t_mae <- c(data$training_mae[index], t_mae)
  t_medae <- c(data$training_medae[index], t_medae)
  t_rmse <- c(data$training_rmse[index], t_rmse)
  t_ssim <- c(data$training_ssim[index], t_ssim)
  t_zncc <- c(data$training_zncc[index], t_zncc)
  
  v_dicesb <- c(data$validation_loss[index], v_dicesb)
  v_mae <- c(data$validation_mae[index], v_mae)
  v_medae <- c(data$validation_medae[index], v_medae)
  v_rmse <- c(data$validation_rmse[index], v_rmse)
  v_ssim <- c(data$validation_ssim[index], v_ssim)
  v_zncc <- c(data$validation_zncc[index], v_zncc)
  
}

for(sub in subs){
  dir <- paste0("test_metrics/", sub, "_", version, ".txt")
  
  data <- readLines(dir)
  
  f_dicesb <- c(as.numeric(data[1]), f_dicesb)
  f_mae <- c(as.numeric(data[2]), f_mae)
  f_medae <- c(as.numeric(data[3]), f_medae)
  f_rmse <- c(as.numeric(data[4]), f_rmse)
  f_ssim <- c(as.numeric(data[5]), f_ssim)
  f_zncc <- c(as.numeric(data[6]), f_zncc)
  f_comp <- c(as.numeric(data[7]), f_comp)
  f_corr <- c(as.numeric(data[8]), f_corr)
  f_qual <- c(as.numeric(data[8]), f_qual)
  
}


t_dicesb = normalize(t_dicesb)
t_mae = normalize(t_mae)
t_medae = normalize(t_medae)
t_rmse = normalize(t_rmse)
t_ssim = normalize(t_ssim)
t_zncc = normalize(t_zncc)

v_dicesb = normalize(v_dicesb)
v_mae = normalize(v_mae)
v_medae = normalize(v_medae)
v_rmse = normalize(v_rmse)
v_ssim = normalize(v_ssim)
v_zncc = normalize(v_zncc)

f_dicesb = normalize(f_dicesb)
f_mae = normalize(f_mae)
f_medae = normalize(f_medae)
f_rmse = normalize(f_rmse)
f_ssim = normalize(f_ssim)
f_zncc = normalize(f_zncc)
f_comp = normalize(f_comp)
f_corr = normalize(f_corr)
f_qual = normalize(f_qual)



t_mean_dicesb = mean(t_dicesb)
t_mean_mae = mean(t_mae)
t_mean_medae = mean(t_medae)
t_mean_rmse = mean(t_rmse)
t_mean_ssim = mean(t_ssim)
t_mean_zncc = mean(t_zncc)

v_mean_dicesb = mean(v_dicesb)
v_mean_mae = mean(v_mae)
v_mean_medae = mean(v_medae)
v_mean_rmse = mean(v_rmse)
v_mean_ssim = mean(v_ssim)
v_mean_zncc = mean(v_zncc)

t_std_dicesb = sd(t_dicesb)
t_std_mae = sd(t_mae)
t_std_medae = sd(t_medae)
t_std_rmse = sd(t_rmse)
t_std_ssim = sd(t_ssim)
t_std_zncc = sd(t_zncc)

v_std_dicesb = sd(v_dicesb)
v_std_mae = sd(v_mae)
v_std_medae = sd(v_medae)
v_std_rmse = sd(v_rmse)
v_std_ssim = sd(v_ssim)
v_std_zncc = sd(v_zncc)


f_mean_dicesb = mean(f_dicesb)
f_mean_mae = mean(f_mae)
f_mean_medae = mean(f_medae)
f_mean_rmse = mean(f_rmse)
f_mean_ssim = mean(f_ssim)
f_mean_zncc = mean(f_zncc)
f_mean_comp = mean(f_comp)
f_mean_corr = mean(f_corr)
f_mean_qual = mean(f_qual)

f_std_dicesb = sd(f_dicesb)
f_std_mae = sd(f_mae)
f_std_medae = sd(f_medae)
f_std_rmse = sd(f_rmse)
f_std_ssim = sd(f_ssim)
f_std_zncc = sd(f_zncc)
f_std_comp = sd(f_comp)
f_std_corr = sd(f_corr)
f_std_qual = sd(f_qual)




standarise <- function(x, m, s){
  return((x - m) / s)
}





firstnet <- "denseunet"
secondnet <- "aspp"



dir1 <- paste0("runs_5e-05/seed/final/", firstnet, "/", version, "/progress.csv")
dir2 <- paste0("runs_5e-05/seed/final/", secondnet, "/", version, "/progress.csv")

dir3 <- paste0("test_metrics/", firstnet, "_", version, ".txt")
dir4 <- paste0("test_metrics/", secondnet, "_", version, ".txt")

data1 <- read.csv(dir1)
data2 <- read.csv(dir2)
data3 <- readLines(dir3)
data4 <- readLines(dir4)

index1 <- which.min(data1$validation_loss)
index2 <- which.min(data2$validation_loss)



s = standarise(data1$training_loss[index1], t_mean_dicesb, t_std_dicesb)
f = standarise(data2$training_loss[index2], t_mean_dicesb, t_std_dicesb)
t_improvement_loss = (s - f) / abs(s)

s = standarise(data1$training_mae[index1], t_mean_dicesb, t_std_mae)
f = standarise(data2$training_mae[index2], t_mean_dicesb, t_std_mae)
t_improvement_mae = (s - f) / abs(s)


s = standarise(data1$training_medae[index1], t_mean_dicesb, t_std_medae)
f = standarise(data2$training_medae[index2], t_mean_dicesb, t_std_medae)
t_improvement_medae = (s - f) / abs(s)

s = standarise(data1$training_rmse[index1], t_mean_dicesb, t_std_rmse)
f = standarise(data2$training_rmse[index2], t_mean_dicesb, t_std_rmse)
t_improvement_rmse = (s - f) / abs(s)

s = standarise(data1$training_ssim[index1], t_mean_dicesb, t_std_ssim)
f = standarise(data2$training_ssim[index2], t_mean_dicesb, t_std_ssim)
t_improvement_ssim = (s - f) / abs(s) * -1

s = standarise(data1$training_zncc[index1], t_mean_dicesb, t_std_zncc)
f = standarise(data2$training_zncc[index2], t_mean_dicesb, t_std_zncc)
t_improvement_zncc = (s - f) / abs(s) * -1


print(cat(
  " Improvement training DICEsb: ", t_improvement_loss, "\n",
  "Improvement training MAE   : ", t_improvement_mae, "\n",
  "Improvement training MedAE : ", t_improvement_medae, "\n",
  "Improvement training RMSE  : ", t_improvement_rmse, "\n",
  "Improvement training SSIM  : ", t_improvement_ssim, "\n",
  "Improvement training ZNCC  : ", t_improvement_zncc, "\n"
))
print(paste0("Final training improvement = ", mean(
  c(t_improvement_loss, 
    t_improvement_mae,
    t_improvement_medae,
    t_improvement_rmse,
    t_improvement_ssim,
    t_improvement_zncc
  )
)))



s = standarise(data1$validation_loss[index1], v_mean_dicesb, v_std_dicesb)
f = standarise(data2$validation_loss[index2], v_mean_dicesb, v_std_dicesb)
v_improvement_loss = (s - f) / abs(s)

s = standarise(data1$validation_mae[index1], v_mean_dicesb, v_std_mae)
f = standarise(data2$validation_mae[index2], v_mean_dicesb, v_std_mae)
v_improvement_mae = (s - f) / abs(s)


s = standarise(data1$validation_medae[index1], v_mean_dicesb, v_std_medae)
f = standarise(data2$validation_medae[index2], v_mean_dicesb, v_std_medae)
v_improvement_medae = (s - f) / abs(s)

s = standarise(data1$validation_rmse[index1], v_mean_dicesb, v_std_rmse)
f = standarise(data2$validation_rmse[index2], v_mean_dicesb, v_std_rmse)
v_improvement_rmse = (s - f) / abs(s)

s = standarise(data1$validation_ssim[index1], v_mean_dicesb, v_std_ssim)
f = standarise(data2$validation_ssim[index2], v_mean_dicesb, v_std_ssim)
v_improvement_ssim = (s - f) / abs(s) * -1

s = standarise(data1$validation_zncc[index1], v_mean_dicesb, v_std_zncc)
f = standarise(data2$validation_zncc[index2], v_mean_dicesb, v_std_zncc)
v_improvement_zncc = (s - f) / abs(s) * -1


print(cat(
  " Improvement validation DICEsb: ", v_improvement_loss, "\n",
  "Improvement validation MAE   : ", v_improvement_mae, "\n",
  "Improvement validation MedAE : ", v_improvement_medae, "\n",
  "Improvement validation RMSE  : ", v_improvement_rmse, "\n",
  "Improvement validation SSIM  : ", v_improvement_ssim, "\n",
  "Improvement validation ZNCC  : ", v_improvement_zncc, "\n"
))
print(paste0("Final validation improvement = ", mean(
  c(v_improvement_loss, 
    v_improvement_mae,
    v_improvement_medae,
    v_improvement_rmse,
    v_improvement_ssim,
    v_improvement_zncc
  )
)))




s = standarise(as.numeric(data3[1]), f_mean_dicesb, f_std_dicesb)
f = standarise(as.numeric(data4[1]), f_mean_dicesb, f_std_dicesb)
f_improvement_loss = (s - f) / abs(s)

s = standarise(as.numeric(data3[2]), f_mean_dicesb, f_std_mae)
f = standarise(as.numeric(data4[2]), f_mean_dicesb, f_std_mae)
f_improvement_mae = (s - f) / abs(s)

s = standarise(as.numeric(data3[3]), f_mean_dicesb, f_std_medae)
f = standarise(as.numeric(data4[3]), f_mean_dicesb, f_std_medae)
f_improvement_medae = (s - f) / abs(s)

s = standarise(as.numeric(data3[4]), f_mean_dicesb, f_std_rmse)
f = standarise(as.numeric(data4[4]), f_mean_dicesb, f_std_rmse)
f_improvement_rmse = (s - f) / abs(s)

s = standarise(as.numeric(data3[5]), f_mean_dicesb, f_std_ssim)
f = standarise(as.numeric(data4[5]), f_mean_dicesb, f_std_ssim)
f_improvement_ssim = (s - f) / abs(s) * -1

s = standarise(as.numeric(data3[6]), f_mean_dicesb, f_std_zncc)
f = standarise(as.numeric(data4[6]), f_mean_dicesb, f_std_zncc)
f_improvement_zncc = (s - f) / abs(s) * -1

s = standarise(as.numeric(data3[7]), f_mean_dicesb, f_std_comp)
f = standarise(as.numeric(data4[7]), f_mean_dicesb, f_std_comp)
f_improvement_comp = (s - f) / abs(s) * -1

s = standarise(as.numeric(data3[8]), f_mean_dicesb, f_std_corr)
f = standarise(as.numeric(data4[8]), f_mean_dicesb, f_std_corr)
f_improvement_corr = (s - f) / abs(s) * -1

s = standarise(as.numeric(data3[9]), f_mean_dicesb, f_std_qual)
f = standarise(as.numeric(data4[9]), f_mean_dicesb, f_std_qual)
f_improvement_qual = (s - f) / abs(s) * -1

print(cat(
  " Improvement testing DICEsb: ", f_improvement_loss, "\n",
  "Improvement testing MAE   : ", f_improvement_mae, "\n",
  "Improvement testing MedAE : ", f_improvement_medae, "\n",
  "Improvement testing RMSE  : ", f_improvement_rmse, "\n",
  "Improvement testing SSIM  : ", f_improvement_ssim, "\n",
  "Improvement testing ZNCC  : ", f_improvement_zncc, "\n",
  "Improvement testing COMP  : ", f_improvement_comp, "\n",
  "Improvement testing CORR  : ", f_improvement_corr, "\n",
  "Improvement testing QUAL  : ", f_improvement_qual, "\n"
))
print(paste0("Final testing improvement = ", mean(
  c(f_improvement_loss, 
    f_improvement_mae,
    f_improvement_medae,
    f_improvement_rmse,
    f_improvement_ssim,
    f_improvement_zncc,
    f_improvement_comp,
    f_improvement_corr,
    f_improvement_qual
  )
)))




    
    
