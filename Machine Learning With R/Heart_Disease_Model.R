# Tips to get started
## To install needed CRAN packages:
library(tidyverse)
library(Hmisc)
library(GGally)
library(caret)
library(gmodels)
library(rpart)
library(rpart.plot)
library(dendextend)
library(randomForest)
library(mlr3)
library(devtools)
library(corrplot)

## To install needed Bioconductor packages:
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c("limma", "edgeR"))

# To install libraries from GitHub source
library(devtools)
install_github("vqv/ggbiplot")

# Import Data 
heart_Data <- read.csv("~/VSC/Git_/AI/Machine Learning With R/Data/Heart_Disease.csv")
describe(heart_Data)

#Exploratory Data Analysis
heart_Data$sex[heart_Data$sex== 1] <- "Male"
heart_Data$sex[heart_Data$sex== 0] <- "Female"

# Data Visualization 
library(corrplot)

heart_Data %>% ggplot(aes(fasting.blood.sugar, fill= cholesterol))+
  geom_bar()+
  facet_wrap(~chest.pain.type)

heart_Data %>% ggplot(aes(cholesterol))+
  geom_boxplot()+
  facet_wrap(~chest.pain.type)

data_cor <- cor(heart_Data[,1:13])
corrplot(data_cor,method = "square")

ggpairs(heart_Data[3:0], aes(color=, alpha=0.4))

fitted(heart_Data)

ppv <- preProcess(heart_Data, method = c("center", "scale"))
heart_Data_tr <- predict(ppv, heart_Data)