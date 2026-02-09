library(raster)
library(ggplot2)
library(vegan)
library(dplyr)
library(tibble)
library(tidyr)
library(viridis)
library(patchwork)
library(rlang)
library(purrr)


# Working directory should be root of this project

# Load base raster
base_raster <- raster("data/maps/SR_all_trees_observed.tif")
template_raster <- init(base_raster, "cell")
values(template_raster) <- NA
names(template_raster) <- "Grid"

# Load metrics CSV
metrics_path <- "output/spatial_analysis/REV_obs_results_200_extended.csv"
all_metrics <- read.csv(metrics_path)

#### (To double-check simpsons index) #####
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


plot_two_rasters <- function(r1, r2,
                             title1 = "Top Map", title2 = "Bottom Map",
                             palette1 = viridis::viridis(20),
                             palette2 = viridis::viridis(20),
                             save_path = NULL) {
  # if (!is.null(save_path)) {
  #   png(save_path, width = 6.3, height = 4, units = "in", res = 300)
  # }
  
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

  
  # Save the plot if requested
  if (!is.null(save_path)) {
    # Record the current plot
    p <- recordPlot()
    
    # Open PNG device and replay the recorded plot
    png(save_path, width = 6.3, height = 4, units = "in", res = 300)
    replayPlot(p)
    #dev.off()
    
    message("Saved to: ", save_path)
  
  }
}

plot_two_rasters(raster_list[['nspec']], raster_list[['nclust']],
                 'Num. of Species', 'Num. of Clusters',
                 #save_path = NULL
                 save_path = 'output/figures/nspec_nclust_map.png'
                 )

plot_two_rasters(log(raster_list[['nspec']]), raster_list[['nclust']],
                 'Log Num. of Species', 'Num. of Clusters',
                 save_path = NULL
                 #save_path = 'output/figures/nspec_nclust_map.png'
)



plot_two_rasters(raster_list[['norder']], raster_list[['nclust']],
                 'Num. of Taxonomic Groups \n (Order)', 'Num. of Functional Groups',
                 #save_path = NULL
                 save_path = 'output/figures/norder_nclust_map.png'
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

plot_metric_scatter(all_metrics, "nspec", "cluster_simpson",
                    log_transform = FALSE, add_lm = FALSE,
                    #x_label = 'Number of Species', y_label = "Inverse Simpson Index",
                    save_path = 'output/figures/nspec_vs_isi.pdf')

all_metrics$cluster_gini <- 1-all_metrics$cluster_simpson

1/(all_metrics[['simpson']])

all_metrics$inverse_simpson

### S = sum 1/p2_i

summary(lm(cluster_simpson~fdr*raoq*nspec, data = all_metrics))

summary(lm(fdr~raoq, data = all_metrics))

par(
  mfrow = c(2,1)
)

plot_metric_scatter(all_metrics, "nspec", "cluster_simpson",
                    log_transform = FALSE, add_lm = FALSE,
                    #x_label = 'Number of Species', y_label = "Inverse Simpson Index",
                    save_path = 'output/figures/nspec_vs_isi.pdf')




facet_scatter_by_x <- function(data, x_metrics, y_metric,
                               log_transform = FALSE, add_lm = FALSE,
                               x_labels = NULL, facet_labels = NULL,
                               ncol = NULL, nrow = NULL,
                               save_path = NULL, width = 8, height = 6, dpi = 300) {
  
  # Reshape data to long format for faceting
  plot_data <- x_metrics %>%
    map_dfr(~ {
      df <- data %>%
        select(x = all_of(.x), y = all_of(y_metric)) %>%
        mutate(x_var = .x)
      colnames(df)[1:2] <- c("x", "y")
      df
    })
  
  # Apply log transform if needed
  if (log_transform) {
    plot_data <- plot_data %>%
      mutate(x = log1p(x), y = log1p(y))
  }
  
  # Label mapping
  plot_data <- plot_data %>%
    mutate(x_var = factor(x_var, levels = x_metrics,
                          labels = facet_labels %||% x_labels %||% x_metrics))
  
  # Base plot
  p <- ggplot(plot_data, aes(x = x, y = y)) +
    # geom_bin2d(bins = 100) +
    # stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    # scale_x_continuous(expand = c(0, 0)) +
    # scale_y_continuous(expand = c(0, 0)) +
    theme_bw() +
    geom_point(color = "deepskyblue4", alpha = 0.3, size = 0.8) +
    # labs(x = NULL, y = if (log_transform) paste0("log(1 + ", y_metric, ")") else y_metric) +
    # theme_classic(base_size = 12) +
    facet_wrap(~ x_var, scales = "free_x", ncol = ncol, nrow = nrow)
  # 
  
  # p <- ggplot(plot_data, aes(x = x, y = y)) +
  #   stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  #   scale_fill_viridis_c(option = "C", direction = -1) +  # better fill scale
  #   scale_x_continuous(expand = c(0, 0)) +
  #   scale_y_continuous(expand = c(0, 0)) +
  #   labs(x = NULL, y = if (log_transform) paste0("log(1 + ", y_metric, ")") else y_metric) +
  #   theme_bw(base_size = 12) +
  #   facet_wrap(~ x_var, scales = "free_x", ncol = ncol, nrow = nrow)
  
  
  
  
  # Add linear model line if requested
  if (add_lm) {
    p <- p + geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 1)
  }
  
  print(p)
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = width, height = height, dpi = dpi)
    message("Plot saved to: ", save_path)
  }
}

facet_scatter_by_x(all_metrics, c('nspec', 'raoq', 'fdr'), 'cluster_simpson')


# #### 4 metrics
# plot_4metric_scatter <- function(data, metrics, 
#                                  x_label = NULL, y_label = NULL, 
#                                  log_transform = FALSE, add_lm = TRUE, 
#                                  titles = NULL,
#                                  save_path = NULL, width = 6.30045, height = 4, dpi = 300) {
#   
#   if (length(metrics) != 4 || !all(sapply(metrics, length) == 2)) {
#     stop("metrics must be a list of 4 pairs (each pair of x and y metric names).")
#   }
#   
#   plots <- list()
#   
#   for (i in seq_along(metrics)) {
#     x_metric <- metrics[[i]][1]
#     y_metric <- metrics[[i]][2]
#     
#     # Default axis labels
#     xlab <- if (!is.null(x_label[[i]])) x_label[[i]] else if (log_transform) paste0("log(1 + ", x_metric, ")") else x_metric
#     ylab <- if (!is.null(y_label[[i]])) y_label[[i]] else if (log_transform) paste0("log(1 + ", y_metric, ")") else y_metric
#     
#     plot_data <- data
#     if (log_transform) {
#       plot_data[[x_metric]] <- log1p(plot_data[[x_metric]])
#       plot_data[[y_metric]] <- log1p(plot_data[[y_metric]])
#     }
#     
#     p <- ggplot(plot_data, aes_string(x = x_metric, y = y_metric)) +
#       geom_point(color = "deepskyblue4", alpha = 0.3, size = 0.8) +
#       labs(x = xlab, y = ylab) +
#            #, title = if (!is.null(titles)) titles[i] else paste(x_metric, "vs", y_metric)) +
#       theme_classic(base_size = 12)
#     
#     if (add_lm) {
#       p <- p + geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "solid", linewidth = 1)
#     }
#     
#     plots[[i]] <- p
#   }
#   
#   # Arrange plots in 2x2 grid with shared guides
#   combined_plot <- (plots[[1]] | plots[[2]]) / (plots[[3]] | plots[[4]])
#   
#   print(combined_plot)
#   
#   if (!is.null(save_path)) {
#     ggsave(filename = save_path, plot = combined_plot, width = width, height = height, dpi = dpi)
#     message("Plot saved to: ", save_path)
#   }
# }
# 
# 
# plot_4metric_scatter(
#   data = all_metrics,
#   metrics = list(
#     c("raoq", "fdr"),
#     c("cluster_simpson", "fdr"),
#     c("raoq", "cluster_simpson"),
#     c("cluster_simpson", "nspec")
#   ),
#   x_label = c("RaoQ", "Funct. Simp.", "RaoQ", "Funct. Simp."),
#   y_label = c("FDR", "FDR", "Funct. Simp.", "N. Species"),
#   log_transform = FALSE,
#   add_lm = FALSE,
#   titles = c("", "", "", ""),
#   save_path = "output/figures/diversity_metrics.png", dpi = 600
# )

