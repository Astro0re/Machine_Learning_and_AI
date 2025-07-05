library(ggpairs)
library(GGally)

breast_cancer <- read.csv("C:/Users/USER/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data", header=FALSE)

breast_cancer_col_name <- c("ID","Diagnosis","Radius.Mean","Texture.Mean",
                            "Perimeter.Mean","Area.Mean","Smoothness.Mean","Compactness.Mean",
                            "Concavity.Mean","Concave.Points.Mean","Symmetry.Mean",
                            "Fractal.Dimension.Mean","Radius.SE","Texture.SE",
                            "Perimeter.SE","Area.SE","Smoothness.SE","Compactness.SE",
                            "Concavity.SE","Concave.Points.SE","Symmetry.SE",
                            "Fractal.Dimension.SE","Radius.Worst","Texture.Worst",
                            "Perimeter.Worst","Area.Worst","Smoothness.Worst",
                            "Compactness.Worst","Concavity.Worst","Concave.Points.Worst",
                            "Symmetry.Worst","Fractal.Dimension.Worst")

colnames(breast_cancer) <- breast_cancer_col_name

breast_cancer$Diagnosis <- as.factor(breast_cancer$Diagnosis)

breast_cancer$ID <- 0

ggpairs(breast_cancer[2:6], aes(color=Diagnosis, alpha=0.4))

# Remove first column
breastCancerDataNoID <- breastCancerData[2:ncol(breastCancerData)]


ggpairs(breastCancerDataNoID[1:5], aes(color=Diagnosis, alpha=0.4))

library(caret)

# Center & scale data
ppv <- preProcess(breastCancerDataNoID, method = c("center", "scale"))
breastCancerDataNoID_tr <- predict(ppv, breastCancerDataNoID)

# Summarize first 5 columns of the original data
breastCancerDataNoID[1:5] %>% summary()

# Summarize first 5 columns of the re-centered and scaled data
breastCancerDataNoID_tr[1:5] %>% summary()


ggpairs(breastCancerDataNoID_tr[1:5], aes(color=Diagnosis, alpha=0.4))