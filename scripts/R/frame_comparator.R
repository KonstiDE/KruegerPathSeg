library(ggplot2)
library(viridis)
library(stringr)
library(common)

move <- function(from, to) {
  todir <- dirname(to)
  if (!isTRUE(file.info(todir)$isdir)) dir.create(todir, recursive=TRUE)
  file.rename(from = from,  to = to)
}

setwd("E:/")

path_with_sorted_frames <- "frames_buffer/"
path_with_unsorted_frames <- "frames_cluster/"

sub_paths <- c("test", "train", "validation")

for(sub_path in sub_paths){
  path_sort <- paste0(path_with_sorted_frames, sub_path)
  
  sorted_files <- list.files(path_sort, pattern = "*.npz")
  
  for(file in sorted_files){
    if(file.exists(paste0(path_with_unsorted_frames, file))){
      move(
        paste0(path_with_unsorted_frames, file),
        paste0(path_with_unsorted_frames, sub_path, "/", file)
      ) 
    }
  }
}