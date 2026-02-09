# Load necessary libraries
library(ggplot2)
library(ggdendro)

# Generate a random distance matrix
set.seed(42)
size <- 10
distance_matrix <- matrix(runif(size * size), nrow = size)
distance_matrix <- (distance_matrix + t(distance_matrix)) / 2
diag(distance_matrix) <- 0

# Perform hierarchical clustering using Ward's method
hc <- hclust(as.dist(distance_matrix), method = "ward.D2")

# Convert to dendrogram object
dend <- as.dendrogram(hc)

# Create a data frame for plotting
dend_data <- dendro_data(dend, type = "rectangle")

# Create a function to convert Cartesian coordinates to polar coordinates
cartesian_to_polar <- function(data) {
  max_y <- max(data$segment$y)
  data$segment <- transform(data$segment,
                            angle = atan2(y, yend),
                            r = sqrt(y^2 + yend^2))
  data$segment$y <- data$segment$r
  data$segment$yend <- data$segment$r
  return(data)
}

# Convert the dendrogram data to polar coordinates
dend_data <- cartesian_to_polar(dend_data)

# Plot the circular dendrogram using ggplot2
ggplot() +
  geom_segment(data = dend_data$segment,
               aes(x = x, y = y, xend = xend, yend = yend),
               color = "blue") +
  coord_polar(theta = "x") +
  theme_void() +
  ggtitle("Circular Dendrogram")
