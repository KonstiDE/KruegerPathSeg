
library(terra)
library(BAMMtools)
library(sf)
library(ggplot2)
library(tidyterra)
library(lwgeom)
library(Rfast)
library(smoothr)

normalize <- function(v) {
  return (v / sqrt(sum(v^2)))
}

det_between_vectors <- function(v1, v2) {
  return(v1[1] * v2[2] - v1[2] * v2[1])
}

rotate_points_90_degrees <- function(p1, p2, m, l) {
  rotation_matrix <- matrix(c(0, -1, 1, 0), nrow = 2)
  rotated_p1 <- t(rotation_matrix %*% t(p1 - m))
  rotated_p2 <- t(rotation_matrix %*% t(p2 - m))

  return(rbind(rotated_p1 * l + m, rotated_p2 * l + m))
}

matrix2sfline <- function(m, crs, coords=c("X", "Y")){
  st_union(st_as_sf(data.frame(m), coords=coords, crs=crs)) %>% st_cast("LINESTRING")
}
matrix2sfmultipoint <- function(m, crs, coords=c("X", "Y")){
  st_union(st_as_sf(data.frame(m), coords=coords, crs=crs))
}

split_multilinestrings <- function(sf_object) {
  sf_geom <- st_geometry(sf_object)

  geoms <- NULL
  for (t in seq_along(sf_geom)){
    if (st_geometry_type(sf_geom[t]) == "MULTILINESTRING") {
      df <- as.data.frame(st_coordinates(sf_geom[t]))

      linestrings <- st_as_sfc(lapply(split(df, df$L1), function(df) {
        st_linestring(as.matrix(df[, c("X", "Y")]))
      }))

      geoms <- rbind(geoms, matrix(linestrings))

    } else if (st_geometry_type(sf_geom[t]) == "LINESTRING") {
        geoms <- rbind(geoms, sf_geom[t])
    }
  }

  return(st_sfc(geoms, crs=st_crs(sf_object)))
}

paths <- c(
  "svalbard/"
)

for(subpath in paths){
  rgb <- rast(paste0("E:/master/", subpath, "/rgb_clipped.tif"))

  tracks <- vect(paste0("E:/master/", subpath, "/completelines_clipped.gpkg"))
  tracks_sf <- st_as_sf(tracks)
  tracks_sf <- split_multilinestrings(tracks_sf)

  # standard were grid_size=1 and grid_width_half=5
  
  profile <- 1/0.05
  
  glhs <- c(5)
  ls <- c(1)
  sampling_widths <- c(1)
  reserved_spaces <- c(2)

  for(grid_size in ls){
    for(grid_width_half in glhs){
      for(sampling_width in sampling_widths){
        for(reserved_space in reserved_spaces){
          polygons <- list()
          grids <- list()

          t <- 0
          sf_crs <- st_crs(tracks_sf)
          full_length <- length(tracks_sf)

          pb <- txtProgressBar(min = 0, max = full_length, style = 3, width = 50, char = "=")

          for(t in seq(1, full_length)){
            points <- st_line_sample(tracks_sf[t], density=1)
            points <- as.data.frame(st_coordinates(points))[,1:2]
            points <- suppressWarnings({
              st_as_sf(points, coords=c("X", "Y"), crs=sf_crs)
            })

            points <- rbind(
              as.data.frame(st_startpoint(tracks_sf[t])),
              points,
              as.data.frame(st_endpoint(tracks_sf[t]))
            )

            max_points1 <- list()
            max_points2 <- list()

            for(i in seq(1, nrow(points) - 1, by=1)){
              p1 <- st_coordinates(points[i, ])
              p2 <- st_coordinates(points[i + 1, ])

              m <- p1 + (p2 - p1) / 2
              og_diff <- normalize((p2 - p1)) / 40

              det <- det_between_vectors(as.numeric(og_diff), c(1, 0))

              ortho <- rotate_points_90_degrees(p1, p2, m, grid_size)

              di <- dist(ortho)
              if(di != grid_size && di > 0){
                diff <- m - ortho[1, ]
                f <- grid_size / di

                ortho[1, ] <- m + diff * f
                ortho[2, ] <- m - diff * f
              }
              ortho_sf <- matrix2sfline(ortho, crs=sf_crs)

              lsm <- st_coordinates(st_line_sample(ortho_sf, density = 1 / 0.05))[, c(1, 2)]
              if(det <= 0){
                lsm <- apply(lsm, MARGIN = 2, rev)
              }

              # ggRGB(crop(rgb, ext(vect(ortho_sf)) + c(0.5, 0.5, 0, 0)), r=1, g=2, b=3) +
              #   geom_sf(data = ortho_sf) +
              #   geom_sf(data = st_crop(tracks_sf, ext(vect(ortho_sf)) + c(0.5, 0.5, 0, 0)))
              #   geom_point(data = data.frame(lsm), mapping = aes(X, Y)) +
              #   geom_point(data = data.frame(p1), mapping = aes(X, Y, color = ""))

              lsm_before <- NULL
              lsm_after <- NULL

              for(g in seq(-grid_width_half, -1, by=1)){
                lsm_before <- rbind(lsm_before, eachrow(lsm[1:round(profile * grid_size), ], og_diff * g, "+"))
                lsm_after <- rbind(lsm_after, eachrow(lsm[1:round(profile * grid_size), ], og_diff * (g+ grid_width_half + 1), "+"))
              }

              lsm <- data.frame(rbind(lsm_before, lsm, lsm_after))

              lsm_sf <- matrix2sfmultipoint(lsm, sf_crs)

              colnames(lsm_before) <- c("X", "Y")
              colnames(lsm_after) <- c("X", "Y")

              # ggRGB(crop(rgb, ext(vect(lsm, geom=c("X", "Y"), crs=crs(tracks)))), r=1, g=2, b=3) +
              #   geom_sf(data = ortho_sf) +
              #   geom_point(data = data.frame(rbind(p1, p2)), mapping = aes(X, Y)) +
              #   geom_point(data = data.frame(lsm), mapping = aes(X, Y))

              ortho <- extract(rgb[[1:3]], vect(lsm, geom=c("X", "Y"), crs=crs(tracks)), ID=F)
              ortho <- cbind(ortho, X=lsm$X, Y=lsm$Y)

              ortho$ID <- 0:(nrow(ortho) - 1)

              ortho$rowID <- ortho$ID %% profile
              ortho <- ortho[complete.cases(ortho), ]

              # ggRGB(crop(rgb, ext(vect(ortho, geom=c("X", "Y"), crs=crs(tracks)))), r=1, g=2, b=3) +
              #   geom_point(data = ortho, mapping = aes(X, Y))

              tryCatch({
                ortho_agg <- aggregate.data.frame(ortho, by=list(ortho$rowID), FUN="mean", na.action = na.omit)[, -1]
                
                cluster <- kmeans(ortho_agg[, c(1, 2, 3)], centers = 2)
                ortho_agg$cluster <- cluster$cluster
                
                med_line_sf <- st_as_sf(ortho_agg, coords = c("X", "Y"), crs=sf_crs)
                
                mid_point <- ortho_agg[st_nearest_feature(tracks_sf[t], med_line_sf), ]
                mid_point_id <- mid_point$rowID + 1
                
                # ggRGB(crop(rgb, ext(vect(med_line_sf))), r=1, g=2, b=3) +
                #   geom_sf(data = med_line_sf) +
                #   geom_point(data = ortho_agg, mapping = aes(X, Y, color=cluster))
                
                max_point1 <- NULL
                max_point2 <- NULL
                
                if(mid_point_id - reserved_space > 0){
                  for (a in seq(mid_point_id - reserved_space, 1)){
                    if(ortho_agg[a, ]$cluster != mid_point$cluster){
                      max_point1 <- ortho_agg[a + 1, ]
                      break
                    }
                  }
                }else{
                  max_point1 <- ortho_agg[mid_point_id + reserved_space, ]
                }
                if(is.null(max_point1)){
                  max_point1 <- ortho_agg[reserved_space, ]
                }
                if(mid_point_id + reserved_space < nrow(ortho_agg)){
                  for (a in seq(mid_point_id + reserved_space, nrow(ortho_agg) - 1)){
                    if(ortho_agg[a, ]$cluster != mid_point$cluster){
                      max_point2 <- ortho_agg[a - 1, ]
                      break
                    }
                  }
                }else{
                  max_point2 <- ortho_agg[mid_point_id - reserved_space, ]
                }
                if(is.null(max_point2)){
                  max_point2 <- ortho_agg[nrow(ortho_agg) - reserved_space, ]
                }
                
                max_point1 <- st_as_sf(max_point1, coords = c("X", "Y"), crs = sf_crs)
                max_point2 <- st_as_sf(max_point2, coords = c("X", "Y"), crs = sf_crs)
                
                max_points1[[i]] <- max_point1
                max_points2[[i]] <- max_point2
                
                grids[length(grids) + 1] <- lsm_sf
              }, error = function(cond) {
                  
              })
            }

            max1_line <- do.call(rbind, max_points1)
            max2_line <- do.call(rbind, max_points2)

            pol <- st_polygon(list(rbind(
              matrix(st_coordinates(max1_line), ncol = 2),
              apply(matrix(st_coordinates(max2_line), ncol = 2),2, rev),
              st_coordinates(max1_line)[1, ]
            )), dim = "XY")

            pol <- st_sfc(pol, crs=sf_crs)

            if(!st_is_empty(pol)){
              polygons[t] <- pol
            }

            setTxtProgressBar(pb, t)

          }

          close(pb)

          grids <- st_as_sfc(grids, crs=sf_crs)

          polygons <- st_sfc(polygons, crs=sf_crs)
          polygons_smoothed <- smoothr::smooth(polygons, method = "chaikin")

          st_write(polygons_smoothed, paste0(
            "E:/master/", subpath,
            "polygonized_grid_size", gsub("\\.", "", grid_size),
            "grid_width_half", gsub("\\.", "", grid_width_half),
            "sampling_width", gsub("\\.", "", sampling_width),
            "reserved_space", gsub("\\.", "", reserved_space),
            ".gpkg"
          ), append = FALSE, quiet = TRUE)
          st_write(grids, paste0(
            "E:/master/", subpath,
            "grids_grid_size", gsub("\\.", "", grid_size),
            "grid_width_half", gsub("\\.", "", grid_width_half),
            "sampling_width", gsub("\\.", "", sampling_width),
            "reserved_space", gsub("\\.", "", reserved_space),
            ".gpkg"
          ), append = FALSE, quiet = TRUE)

          print(paste0(
            "Successfully finished configuration on '", subpath,
            "' with grid_size: ", grid_size,
            ", grid_width_half: ", grid_width_half,
            ", sampling_width: ", sampling_width,
            ", reserved_space: ", reserved_space,
            ".gpkg"
          ))
          
        }
      }
    }
  }


}
