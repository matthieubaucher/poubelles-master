library(tools)

renameImgs <- function(dossier, prefix = "img_", start = 1){
  setwd(dossier)
  unique_ext <- unique(file_ext(list.files()))
  
  for (e in 1:length(unique_ext)){
    extension <- unique_ext[e]
    old_files <- list.files(pattern = paste0("\\.", extension)) # Nom des images correspondant à l'extension
    old_files_stems <- gsub(paste0("\\.", extension), "", old_files) # Suppression de l'extension
    new_files_stems <- paste0(prefix, start:(length(old_files_stems) + start - 1)) # Renommage des images (sans l'extension)
    new_files <- paste0(new_files_stems, paste0(".", extension)) # Ajout de l'extension
    file.rename(old_files, new_files) # Renommage des images
    start <- start + length(new_files)
  }
}

addPrefixImgs <- function(pref = "img_"){
  
  unique_ext <- unique(file_ext(list.files()))
  start <- 1
  
  for (e in 1:length(unique_ext)){
    extension <- unique_ext[e]
    old_files <- list.files(pattern = paste0("\\.", extension)) # Nom des images correspondant à l'extension
    old_files_stems <- gsub(paste0("\\.", extension), "", old_files) # Suppression de l'extension
    new_files_stems <- paste0(pref, old_files_stems) # Renommage des images (sans l'extension)
    new_files <- paste0(new_files_stems, paste0(".", extension)) # Ajout de l'extension
    file.rename(old_files, new_files) # Renommage des images
    start <- start + length(new_files)
  }
}

# Bouger les images d'un dossier vers un autre dossier
my_function <- function(x){
  file.rename(
    from = file.path(dossier_old, x),
    to = file.path("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/exemple_kangaroo/kangaroo-master/images_val", x)
  )
}
lapply(lst_val, my_function)

dossier_data <- "C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/data/base_image_train"
setwd("C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/exemple_kangaroo/kangaroo-master/images_train")

addPrefixImgs(pref = "train_")
