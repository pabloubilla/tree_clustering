setwd("C:/Users/pablo/OneDrive/Desktop/tree_traits")
library('MGMM')
library(mice)

# Load the data
df_traits <- read.csv("data/traits_obs_log.csv")

# Make a trait matrix with the traits (not first column)
matrix_traits <- as.matrix(df_traits[, c(2,4,6,7)])
# Perform multiple imputation
imputed_data <- mice(matrix_traits, method = 'pmm', m = 5, seed = 500)
completed_data <- complete(imputed_data, 1)
matrix_traits <- as.matrix(completed_data)

# choose K
choose_k <- ChooseK(
                matrix_traits,
                k0 = 2,
                k1 = NULL,
                boot = 100)



# Fit MGMM
cluster_model <- FitGMM(matrix_traits, 
                      k = 2)

mean(cluster_model)
vcov(cluster_model)


# BIGGER K
cluster_model_v2 <- FitGMM(matrix_traits, 
                      k = 3)

mean(cluster_model_v2)
vcov(cluster_model_v2)


# EVEN BIGGER K
cluster_model_v2 <- FitGMM(matrix_traits, 
                           k = 10)

mean(cluster_model_v2)
vcov(cluster_model_v2)


