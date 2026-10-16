library(ggplot2)
library(viridis)
library(stringr)
library(TTR)

setwd("C:/Users/s371513/Desktop/master_figures/")

path <- "pretests/batch/"


files <- list.files(path, full.names = TRUE)

data <- lapply(files, read.csv)
names(data) <- basename(files)

colors <- c("#ff7f0e", "#1f77b4", "#d62728", "#00d6de")

plot_data <- data.frame()
p <- ggplot()

for(i in seq_along(names(data))){
  name <- names(data)[i]
  
  train_df <- data.frame(
    epoch = seq(1, length(data[[name]]$training_loss)),
    loss = data[[name]]$training_loss,
    dataset = "Training",
    model = paste("Batch-size:", str_split_i(sub(".csv", "", name), "_", 2))
  )
  
  valid_df <- data.frame(
    epoch = seq(1, length(data[[name]]$validation_loss)),
    loss = data[[name]]$validation_loss,
    dataset = "Validation",
    model = paste("Batch-size:", str_split_i(sub(".csv", "", name), "_", 2))
  )
  
  plot_data <- rbind(plot_data, train_df, valid_df)
}

smoothing <- function(x, weight = 0.6, smooth = "loss", cut = "model"){
  models <- unique(x$model)
  
  big_df <- data.frame()
  
  for(m in models){
    
    model_subset <- subset(x, model == m)
    
    v <- seq(1, nrow(model_subset))
    v[1] <- model_subset[smooth][1, ]
    
    for(t in 2:nrow(model_subset)){
      v[t] <- weight * model_subset[smooth][t, ] + (1 - weight) * v[(t - 1)]
    }
    
    df <- model_subset
    df[smooth] <- v
    
    big_df <- rbind(big_df, df)
    
  }
  
  return(big_df)
  
}

training <- subset(plot_data, dataset == "Training")
training_smoothed <- subset(plot_data, dataset == "Training")
training_smoothed <- smoothing(training_smoothed, weight=0.65)

validation <- subset(plot_data, dataset == "Validation")
validation_smoothed <- subset(plot_data, dataset == "Validation")
validation_smoothed <- smoothing(validation_smoothed, weight=0.65)


p <- ggplot(plot_data, aes(x = epoch, y = loss, color = model)) + 
  geom_line(data = training, size = 1.2, alpha = 0.2) +
  geom_line(data = training_smoothed, size = 1.2, alpha = 1) +
  geom_line(data = validation, size = 1.2, alpha = 0.2) +
  geom_line(data = validation_smoothed, size = 1.2, alpha = 1) +
  scale_color_manual(values = rep(colors, 2)) + 
  facet_wrap(~dataset, scales = "free_y") + 
  theme_minimal() +
  theme(
    legend.title = element_blank(),
    legend.position = "right",
    panel.grid.major = element_line(size = 0.5, linetype = 'solid', colour = "grey80"),
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid', colour = "grey90"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  ) +
  labs(
    x = "Epochs", 
    y = "Loss", 
    title = "Training and Validation Loss"
  )

p

ggsave("../../../../Desktop/master_figures/batches.png", p, dpi=600)






