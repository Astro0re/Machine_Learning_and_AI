# Tips to get started
## To install needed CRAN packages:
library(tidyverse)
library(GGally)
library(caret)
library(gmodels)
library(rpart)
library(rpart.plot)
library(dendextend)
library(randomForest)
library(mlr3)
library(devtools)

## To install needed Bioconductor packages:
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c("limma", "edgeR"))

# To install libraries from GitHub source
library(devtools)
install_github("vqv/ggbiplot")

heart_Data <- read.csv("~/VSC/Git_/AI/Machine Learning With R/Data/Heart_Disease.csv")
# My Model

ggpairs(heart_Data, aes(color=, alpha=0.4))

