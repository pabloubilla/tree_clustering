rm(list = ls())

library(mgcv)
library(tidyverse)


## Working directory should be root of this project
## Change as needed
# setwd("~/clustering")

# Load base raster
grids <- read_csv("grid_coordinates.csv")

# read in the functional diversity metrics from Paz et al.
paz_metrics <- read_csv("Paz_et_al_data.csv")

# read in the functional group data for the same grid cells
fg_metrics <- read_csv("Functional_group_results.csv")

# combine
all_metrics <- paz_metrics %>% left_join(fg_metrics)

#### function for calculating the metrics, fit below
fit_gams <- function(data, metrics, x_label = NULL, y_label = NULL) {
    
    
    set.seed(1)
    
    plots <- list()
    
    pcorrs <- all_points <- all_gam <- tibble()
    for (i in seq_along(metrics)) {
        print(i)
        # select the right variables
        x_metric <- metrics[[i]][1]
        y_metric <- metrics[[i]][2]
        xlab <- if (!is.null(x_label[[i]])) x_label[[i]] else if (log_transform) paste0("log(1 + ", x_metric, ")") else x_metric
        ylab <- if (!is.null(y_label[[i]])) y_label[[i]] else if (log_transform) paste0("log(1 + ", y_metric, ")") else y_metric
        plot_data <- data %>% select(Latitude, Longitude, all_of(x_metric), all_of(y_metric)) %>%  filter(complete.cases(.))
        
        if(i > 3){
            # trim out the artifical 1/0 values for monocultures
            plot_data <- plot_data %>% filter(cluster_simpson > 1e-5, cluster_simpson < (1-1e-5))
        }

        # get the overall fit
        f <- as.formula(paste0(y_metric, " ~ s(", x_metric, ", bs = 'bs', k = 7, m = c(2,1)) + s(Longitude, Latitude, bs = 'sos')"))
        g <- gam(f, family = gaussian, data = plot_data, method = "REML", gamma = 1.4)
        
        # get the partial resiudals
        f1 <- as.formula(paste0(x_metric, " ~ s(Longitude, Latitude, bs = 'sos')"))
        f2 <- as.formula(paste0(y_metric, " ~ s(Longitude, Latitude, bs = 'sos')"))
        g1 <- gam(f1, family = gaussian, data = plot_data)
        g2 <- gam(f2, family = gaussian, data = plot_data)
        mycorr <- cor.test(residuals(g1), residuals(g2), method = "pearson")
        pcorrs <- pcorrs %>% bind_rows(tibble(x = metrics[[i]][1], y = metrics[[i]][2], rho = mycorr$estimate, pvalue = mycorr$p.value))
        
 
        # extract the fit and the conf ints
        nd <- setNames(data.frame(seq(min(plot_data[[x_metric]]), max(plot_data[[x_metric]]), length = 200)), x_metric)
        nd$Longitude <- mean(plot_data$Longitude)
        nd$Latitude  <- mean(plot_data$Latitude)
        pr <- predict(g, newdata = nd, se.fit = TRUE, exclude = "s(Longitude,Latitude)")
        spatial_mean <- mean(predict(g, type = "terms")[, "s(Longitude,Latitude)"])
        nd$fit <- pr$fit + spatial_mean
        nd$se  <- pr$se.fit
        nd$x <- nd[[x_metric]]
        nd$y <- nd$fit
        nd$ymin <- nd$fit - 2 * nd$se
        nd$ymax <- nd$fit + 2 * nd$se
        nd$col_var <- xlab
        nd$row_var <- ylab
        
        # combine everything
        plot_data$x <- plot_data[[x_metric]]
        plot_data$y <- plot_data[[y_metric]]
        plot_data$col_var <- xlab
        plot_data$row_var <- ylab
        
        
        # collect into lists, then rbind after the loop
        all_points <- all_points %>% bind_rows(plot_data[, c("x", "y", "col_var", "row_var")])
        all_gam <- all_gam %>% bind_rows(nd[, c("x", "y", "ymin", "ymax", "col_var", "row_var")])

    }
    
    # align the facets
    all_points <- all_points %>%
        mutate(col_var = factor(col_var, levels = unique(x_label)),
               row_var = factor(row_var, levels = unique(y_label)))
    
    all_gam <- all_gam %>%
        mutate(col_var = factor(col_var, levels = unique(x_label)),
               row_var = factor(row_var, levels = unique(y_label)))
    
    # create the plot
    combined <- ggplot(all_points, aes(x, y)) +
        geom_point(color = "navy", alpha = 0.05, size = 0.4) +
        geom_ribbon(data = all_gam, aes(ymin = ymin, ymax = ymax), alpha = 0.2, colour = "darkred", fill = "darkred") +
        geom_line(data = all_gam, colour = "darkred") +
        facet_grid(row_var ~ col_var, scales = "free", switch = "both") +
        labs(x = NULL, y = NULL) +
        theme_classic(base_size = 12) +
        theme(
            strip.background = element_blank(),
            strip.placement = "outside",
            panel.border = element_rect(colour = "black", fill = NA),
            axis.ticks = element_line()
        )

    
    return(list(pcorrs = pcorrs, combined = combined))
}


# fit the models and make the plot
pall <- fit_gams(
    data = all_metrics,
    metrics = list(
        c("nspec", "nclust"),
        c("raoq", "nclust"),
        c("fdr", "nclust"),
        c("nspec", "cluster_simpson"),
        c("raoq", "cluster_simpson"),
        c("fdr", "cluster_simpson")
    ),
    x_label = c("Species Richness", "Rao's Q \n (mean pairwise distance)", "Functional Richness \n (convex hull)", 
                "Species Richness", "Rao's Q \n (mean pairwise distance)", "Functional Richness \n (convex hull)"),
    y_label = c("Functional Group Richness", "Functional Group Richness", "Functional Group Richness", 
                "Functional Redundancy \n (Simpson's Index)", "Functional Redundancy \n (Simpson's Index)", "Functional Redundancy \n (Simpson's Index)")
)


# correlations
pall$pcorrs

# add in letter labels for plotting
p <- pall$combined
gb <- ggplot_build(p)
lay <- gb$layout$layout
tags <- cbind(lay, label = paste0("(", letters[lay$PANEL], ")"), x = -Inf, y = Inf)
p <- p + geom_text(data = tags, aes(x = x, y = y, label = label),
                   hjust = -0.5, vjust = 1.5, fontface = 2, inherit.aes = FALSE)


## output the plot
# ggsave(filename = "metric_grid.png", plot = p, device = "png", dpi = 600, units = "in", width = 7, height = 5) 

