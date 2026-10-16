library(terra)
library(ggplot2)
library(tidyterra)

setwd("E:/master/stolsnek_northern/")

mosaic <- rast("mosaic_denseunet_buffer_clipped_export.tif")
paths <- vect("completelines_clipped.gpkg")

paths <- buffer(paths, 0.1)

# Binarize
mosaic_thresholded <- ifel(mosaic < 0.5, 0, 1)

# Rasterize paths
paths_rasterized <- rasterize(paths, mosaic_thresholded, field=1, background=0)

touched <- mosaic + paths_rasterized


plot(mosaic + paths_rasterized)



