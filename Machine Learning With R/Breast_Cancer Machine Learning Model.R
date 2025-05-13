breast_cancer_stats <- read.csv("C:/Users/USER/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data", header=FALSE)

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

colnames(breast_cancer_stats) <- breast_cancer_col_name

breast_cancer_stats$Diagnosis <- as.factor(breast_cancer_stats$Diagnosis)