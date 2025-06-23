library(raster)
library(ggplot2)
library(vegan)
library(dplyr)
library(tibble)
library(tidyr)
library(viridis)

# Working directory should be root of this project

# Load base raster
base_raster <- raster("data/maps/SR_all_trees_observed.tif")
template_raster <- init(base_raster, "cell")
values(template_raster) <- NA
names(template_raster) <- "Grid"

# Load metrics CSV
metrics_path <- "output/spatial_analysis/REV_obs_results_200_extended.csv"
all_metrics <- read.csv(metrics_path)

# (To double-check simpsons index)
# sites_path <- "output/spatial_analysis/sites_cluster.csv"
# df_sites <- read.csv(sites_path)
# 
# cluster_matrix <- df_sites %>%
#   count(grid_id, cluster) %>%
#   pivot_wider(names_from = cluster, values_from = n, values_fill = 0) %>%
#   column_to_rownames("grid_id")
# 
# cluster_simpson <- diversity(cluster_matrix, index = "simpson")






# Output directory for rasters
output_dir <- "output/spatial_analysis/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# List of column names to rasterize
metric_columns <- colnames(all_metrics)

# Function to create a raster from a metric column
create_metric_raster <- function(column_name, template, data, output_dir) {
  r <- template
  r[as.numeric(data$grid_id)] <- data[[column_name]]
  writeRaster(r, filename = file.path(output_dir, paste0(column_name, ".tif")),
              format = "GTiff", overwrite = TRUE)
  return(r)
}

# Rasterize all metrics and store for plotting
raster_list <- list()
for (column_name in metric_columns) {
  cat("Creating raster for column", column_name, "\n")
  raster_list[[column_name]] <- create_metric_raster(column_name, template_raster, all_metrics, output_dir)
}


# layout(matrix(1:2, ncol = 1), heights = c(1, 1))
par(
  mfrow = c(2,1),
  mai = c(0, 1.2, 0.001, .3),
  oma = c(0, 0, 0, .3)  # no outer margin
  # xaxs = "i", yaxs = "i" # do not expand axis limits
)
# First plot
plot(raster_list[["nclust"]], axes = FALSE, box = FALSE, main = NA)
mtext("Number of Clusters", side = 2, line = 2.5, las = 0)

# Second plot
plot(raster_list[["nspec"]]/raster_list[["nclust"]], axes = FALSE, box = FALSE, main = NA)
mtext("Species per Cluster", side = 2, line = 2.5, las = 0)

plot_two_rasters <- function(r1, r2,
                             title1 = "Top Map", title2 = "Bottom Map",
                             palette1 = viridis::viridis(20),
                             palette2 = viridis::viridis(20),
                             save_path = NULL) {
  if (!is.null(save_path)) {
    png(save_path, width = 6.3, height = 4, units = "in", res = 300)
  }
  
  # adjust margins
  par(
    mfrow = c(2,1),
    mai = c(0.001, 0.3, 0.001, .3), #bottom,left,top,right
    oma = c(0, 0, 0, .3)  
  )
  
  plot(r1, col = palette1, axes = FALSE, box = FALSE)
  mtext(title1, side = 2, line = 0, las = 0)
  plot(r2, col = palette2, axes = FALSE, box = FALSE)
  mtext(title2, side = 2, line = 0, las = 0)
  
  if (!is.null(save_path)) {
    dev.off()
    message("Saved to: ", save_path)
  }
}

plot_two_rasters(raster_list[['nspec']], raster_list[['nclust']],
                 'Num. of Species', 'Num. of Clusters',
                 #save_path = NULL
                 save_path = 'figures/nspec_nclust_map.png'
                 )


# Generalized scatterplot function for any two metrics
plot_metric_scatter <- function(data, x_metric, y_metric, 
                                x_label = NULL, y_label = NULL, 
                                log_transform = FALSE, add_lm = TRUE, 
                                save_path = NULL, width = 6.3, height = 6, dpi = 300) {
  # Handle labels
  xlab <- if (!is.null(x_label)) x_label else if (log_transform) paste0("log(1 + ", x_metric, ")") else x_metric
  ylab <- if (!is.null(y_label)) y_label else if (log_transform) paste0("log(1 + ", y_metric, ")") else y_metric
  
  # Optionally transform the data
  plot_data <- data
  if (log_transform) {
    plot_data[[x_metric]] <- log1p(plot_data[[x_metric]])
    plot_data[[y_metric]] <- log1p(plot_data[[y_metric]])
  }
  
  # Build ggplot
  p <- ggplot(plot_data, aes_string(x = x_metric, y = y_metric)) +
    geom_point(color = "deepskyblue4", alpha = 0.3, size = 0.8) +
    labs(x = xlab, y = ylab) +
    theme_classic(base_size = 12) +
  
  # Add linear model fit
  if (add_lm) {
    p <- p + geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "solid", linewidth = 1)
  }
  
  # Print plot
  print(p)
  
  # Save if path is provided
  if (!is.null(save_path)) {
    ggsave(filename = save_path, plot = p, width = width, height = height, dpi = dpi)
    message("Plot saved to: ", save_path)
  }
}

plot_metric_scatter(all_metrics, "nspec", "inverse_simpson",
                    log_transform = FALSE, add_lm = FALSE,
                    x_label = 'Number of Species', y_label = "Inverse Simpson Index",
                    save_path = 'figures/nspec_vs_isi.pdf')




