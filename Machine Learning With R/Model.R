# Tips to get started
## To install needed CRAN packages:
install.packages("tidyverse")
install.packages("GGally")
install.packages("caret")
install.packages("gmodels")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("dendextend")
install.packages("randomForest")
install.packages("mlr3")
install.packages("devtools")

## To install needed Bioconductor packages:
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c("limma", "edgeR"))

# To install libraries from GitHub source
library(devtools)
install_github("vqv/ggbiplot")


# My Model
library(deepnet)