library(terra)
library(sf)
library(lwgeom)
library(ggplot2)
library(dplyr)
library(tidyr)
library(pals)
library(ggpubr)


setwd("E:/master/stolsnek/part8/")

field <- "angle"

truth <- st_as_sf(vect("angle_ground_truth_merged.gpkg"))
truth[[field]] <- ifelse(truth[[field]] < 0, truth[[field]] + pi, ifelse(truth[[field]] > pi / 2, truth[[field]] - pi / 2, truth[[field]]))
truth[[field]] <- truth[[field]] * 180 / pi

pred <- st_as_sf(vect("angle_prediction_merged.gpkg"))
pred[[field]] <- ifelse(pred[[field]] < 0, pred[[field]] + pi, ifelse(pred[[field]] > pi / 2, pred[[field]] - pi / 2, pred[[field]]))
pred[[field]] <- pred[[field]] * 180 / pi

truth <- st_drop_geometry(truth)
pred <- st_drop_geometry(pred)

angle_bins_truth <- truth %>%
  mutate(angle_bin_truth = cut(angle, breaks = seq(0, 180, by = 10), include.lowest = TRUE)) %>%
  count(angle_bin_truth, name="count_t") %>%
  complete(angle_bin_truth, fill = list(count_t = 0)) %>%
  mutate(percentage_truth = count_t / max(count_t))

angle_bins_pred <- pred %>%
  mutate(angle_bin_pred = cut(angle, breaks = seq(0, 180, by = 10), include.lowest = TRUE)) %>%
  count(angle_bin_pred, name="count_p") %>%
  complete(angle_bin_pred, fill = list(count_p = 0)) %>%
  mutate(percentage_pred = count_p / max(count_p))


angle_bins_combined <- cbind(angle_bins_truth, angle_bins_pred)
angle_bins_combined$diff <- angle_bins_combined$percentage_truth - angle_bins_combined$percentage_pred






gg_truth <- ggplot(angle_bins_truth, aes(x = angle_bin_truth, y = count_t)) +
  geom_bar(stat = "identity", width = 1, fill = "skyblue", color = "black") +
  coord_radial(start = -pi/2, end = pi/2) +
  labs(title = "Wind-rose Plot (Ground Truth)", x = "Count", y = NULL) +
  theme_minimal() + 
  theme(
    axis.text.x = element_text(margin = margin(t = 5)),
    axis.title.x = element_text(hjust = 0.275),
    plot.title = element_text(hjust = 0.5)
  )

gg_prediction <- ggplot(angle_bins_pred, aes(x = angle_bin_pred, y = count_p)) +
  geom_bar(stat = "identity", width = 1, fill = "orange", color = "black") +
  coord_radial(start = -pi/2, end = pi/2) +
  labs(title = "Wind-rose Plot (Prediction)", x = "Count", y = NULL) +
  theme_minimal() + 
  theme(
    axis.text.x = element_text(margin = margin(t = 5)),
    axis.title.x = element_text(hjust = 0.275),
    plot.title = element_text(hjust = 0.5)
  )



ggplot(angle_bins_combined) +
  geom_bar(aes(x = angle_bin_pred, y = percentage_pred, fill = diff), 
           stat = "identity", width = 1, color = "black", alpha = 0.7) +
  
  geom_bar(aes(x = angle_bin_truth, y = percentage_truth, fill = diff), 
           stat = "identity", width = 1, color = "black", alpha = 0.7) +
  
  coord_radial(start = -pi/2, end = pi/2) + 
  scale_fill_gradient2(
    low = coolwarm(n=25)[25],
    mid = "white",
    high = coolwarm(n=25)[1],
    midpoint = 0,
    guide = guide_colorbar(direction = "horizontal", title.position = "top")
  ) +
  labs(title = "Wind-Rose Comparison of Ground Truth and Prediction", 
       x = "Normalized amount", y = NULL, fill = "Difference") + 
  theme_minimal() +
  theme(
    axis.text.x = element_text(margin = margin(t = 5)),
    axis.title.x = element_text(hjust = 0.275),
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0),
    legend.box.margin = margin(t = -60, r = -450)
  )


ggarrange(gg_truth, gg_prediction, nrow = 1, ncol = 2)


ggsave("windrose_merged_diff.pdf", dpi=600)



# angle: atan2(y(end_point($geometry)) - y(start_point($geometry)), x(end_point($geometry)) - x(start_point($geometry)))
# angle_inv: atan2(x(end_point($geometry)) - x(start_point($geometry)), y(end_point($geometry)) - y(start_point($geometry)))
