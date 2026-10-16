library(terra)

setwd("E:/master/koger/riverside10/")


# Calculate maximums 
versions <- c("cluster", "buffer")
paths <- c("completelines_clipped_buffered.gpkg", "polygons.gpkg")
subs <- c("unet", "denseunet", "aspp", "attention", "stacked")

for(version in versions){
  for(sub in subs){
    mosaic <- rast(paste0("mosaic_", sub, "_", version, ".tif"))
    
    if(version == "buffer"){
      path <- "completelines_clipped_buffered.gpkg"
    }else{
      path <- "polygons.gpkg"
    }
    
    trails <- vect(path)
    trails <- terra::makeValid(trails)
    
    trails_outer <- terra::buffer(trails, width=0.25)
    
    needs_to_be_one <- terra::mask(mosaic, trails)
    needs_to_be_zero <- terra::mask(mosaic, trails_outer - trails)
    
    values_inner <- values(needs_to_be_one, na.rm=T)
    values_outer <- values(needs_to_be_zero, na.rm=T)
    
    accuracy_inner <- sum(values_inner) / length(values_inner)
    accuracy_outer <- sum(values_outer) / length(values_outer)
    
    print(paste0("LBA: (inner/outer)", round(accuracy_inner, 4), "/", round(accuracy_outer, 4), ", config: ", version, " and ", sub))
  }
}


mosaic <- rast("mosaic_denseunet_buffer_clipped_export.tif")
path <- "completelines_clipped.gpkg"


trails <- vect(path)
trails <- buffer(trails, 0.25)
trails <- terra::makeValid(trails)

trails_outer <- terra::buffer(trails, width=0.25)

needs_to_be_one <- terra::mask(mosaic, trails)
needs_to_be_zero <- terra::mask(mosaic, trails_outer - trails)

values_inner <- values(needs_to_be_one, na.rm=T)
values_outer <- values(needs_to_be_zero, na.rm=T)

accuracy_inner <- sum(values_inner) / length(values_inner)
accuracy_outer <- sum(values_outer) / length(values_outer)


