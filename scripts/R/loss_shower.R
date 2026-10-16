setwd("E:/master/ray_results/")

dir <- "runs_5e-05/seed/final/stacked/cluster/progress.csv"

data <- read.csv(dir)

index <- which.min(data$validation_loss)

cat(noquote(paste0(
  "$", round(data$training_loss[index], 4), "$\n",
  "$", round(data$training_mae[index], 4), "$\n",
  "$", round(data$training_medae[index], 4), "$\n",
  "$", round(data$training_rmse[index], 4), "$\n",
  "$", round(data$training_ssim[index], 4), "$\n",
  "$", round(data$training_zncc[index], 4), "$\n",
  "\n"
)))

cat(noquote(paste0(
  "$", round(data$validation_loss[index], 4), "$\n",
  "$", round(data$validation_mae[index], 4), "$\n",
  "$", round(data$validation_medae[index], 4), "$\n",
  "$", round(data$validation_rmse[index], 4), "$\n",
  "$", round(data$validation_ssim[index], 4), "$\n",
  "$", round(data$validation_zncc[index], 4), "$\n",
  "\n"
)))