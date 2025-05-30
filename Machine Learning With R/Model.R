# Tips to get started
## To install needed CRAN packages:
library(tidyverse)
library(GGally)
library(caret)
library(gmodels)
]library(rpart)
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


# My Model
library(deepnet)
Data <- read.csv()

Data %>%  ggplot(aes= , ) +
  geom.()

