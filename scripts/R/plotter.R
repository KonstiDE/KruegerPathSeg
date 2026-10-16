library(ggplot2)
library(viridis)
library(stringr)

setwd("C:/Users/s371513/Desktop/master_figures/")

path <- "pretests/weight_decay_0/"


files <- list.files(path, full.names = TRUE)

data <- lapply(files, read.csv)
names(data) <- basename(files)

colors <- c("#FFB000", "#FE6100", "#DC267F", "#648FFF")

plot_data <- data.frame()
p <- ggplot()

for(i in seq_along(names(data))){
  name <- names(data)[i]
  
  train_df <- data.frame(
    epoch = seq(1, length(data[[name]]$training_loss)),
    loss = data[[name]]$training_loss,
    dataset = "Training",
    model = paste("Learning rate:", str_split_i(name, "_", 2))
  )
  
  valid_df <- data.frame(
    epoch = seq(1, length(data[[name]]$validation_loss)),
    loss = data[[name]]$validation_loss,
    dataset = "Validation",
    model = paste("Learning rate:", str_split_i(name, "_", 2))
  )
  
  plot_data <- rbind(plot_data, train_df, valid_df)
}



p <- ggplot(plot_data, aes(x = epoch, y = loss, color = model)) + 
  geom_line(data = subset(plot_data, dataset == "Training"), size = 1.2, alpha = 1) +
  geom_line(data = subset(plot_data, dataset == "Validation"), size = 1.2, alpha = 0.2) +
  scale_color_manual(values = rep(colors, 2)) + 
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
    title = "Training and Validation Loss, weight decay: 0"
  )

p

ggsave("../../../Desktop/master_figures/weight_decay_0.png", p, dpi=600)

