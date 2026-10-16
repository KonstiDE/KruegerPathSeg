sumtif <- terra::rast("E:/master/stolsnek_northern/sum_connected.tif")

sumtif <- terra::ifel(sumtif < 0, 0, sumtif)

terra::writeRaster(sumtif, "sum_connected_int.tif", datatype='INT2U', progress=1)
