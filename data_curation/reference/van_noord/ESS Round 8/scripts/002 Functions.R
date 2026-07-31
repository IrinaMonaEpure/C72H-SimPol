#### Functions for binomial regressions ----
# Function to format output
formatModel.glm <- function(x) {
  model <- broom::tidy(x) %>%
    select(-statistic) %>%
    add_row(term = "N", estimate = length(x$residuals)) %>%
    add_row(term = "Deviance", estimate = x$deviance)
  attr(model, "meta_rows") <- c((nrow(model)-1):nrow(model))
  colnames(model) <- c("var", "b", "se", "p")
  return(model)
}

# Function to combine output
combineModels.glm <- function(models) {
  # Join together
  for (i in seq_along(models)) 
    if (i == 1) mat = models[[i]] else
      mat <- full_join(mat, models[[i]], by = c("term"), copy = T)
    
    # Relabel columns and write to file
    rowMeta1 <- nrow(models[[1]])-1
    rowMeta2 <- nrow(models[[1]])
    mat <- rbind(mat[-c(rowMeta1:rowMeta2), ], mat[c(rowMeta1:rowMeta2), ])
    rownames(mat) <- NULL
    colnames(mat) <- c("var", rep(c("b", "se", "p"), times = length(models)))
    
    return(mat)
}



#### Corrplot with two classes, function doublecorrplot ----
doublecorrplot <- function(class1, class2, limits = c(-1,1), labels = rownames(class1)) {
  # Get 'coordinates' of correlations in matrix to be used as the position in the plot
  x <- as.vector(col(class1))
  y <- as.vector(row(class1)*-1)

  ### Upper triangle
  # Class 1 (on left = pi)
  class1_list <- as.vector(class1)
  class1_list <- cbind("value" = class1_list, "value_abs" = abs(class1_list), "start" = pi, x, y)

  # Class 2 (on right = 0)
  class2_list <- as.vector(class2)
  class2_list <- cbind("value" = class2_list, "value_abs" = abs(class2_list), "start" = 0, x, y)
  
  # Merge
  data_upper <- as.data.frame(rbind(class1_list, class2_list))
  data_upper <- data_upper[as.vector(upper.tri(class1)), ]  
  data_upper$value_abs <- scales:::rescale(data_upper$value_abs, c(0.15, 0.45)) # rescale
  
  ### Lower triangle
  diff <- as.vector(class1 - class2)
  data_lower <- as.data.frame(cbind(value = diff, value_abs = abs(diff), x, y))
  data_lower$value_abs <- scales:::rescale(abs(data_lower$value_abs), c(0.15,0.45), # rescale 
                                           from = range(abs(data_upper$value), na.rm = TRUE, finite = TRUE)) # to same scale as upper triangle
  data_lower <- data_lower[as.vector(lower.tri(class1)), ]  
  
  # Crosses 0?
  cross0 <- ifelse(as.vector(class1) > 0 & 0 > as.vector(class2) | as.vector(class2) > 0 & 0 > as.vector(class1), 1, NA_integer_)
  data_lower2 <- as.data.frame(cbind(cross0, x, y))
  data_lower2 <- data_lower2[as.vector(lower.tri(class1)), ]
  data_lower2 <- data_lower2[!is.na(data_lower2$cross0), ]  
  
  # Define colors (similar palette as corrplot)
  gcolors <- c("#67001F", "#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
               "#FFFFFF", 
               "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061")
  
  # Color limits set to max and min correlations?
  if (limits == "max") {
    limits <- rbind(grps$average1, grps$average2)
    limits <- ceiling(max(abs(limits), na.rm=T)*10)/10
    limits <- c(-limits, limits)
  }
  
  # Get number of correlations for the breaks in graph
  n_corr <- nrow(class1)
  
  # Plot the graph
  plot_twoclass <- ggplot(data_upper) + 
    ggforce::geom_arc_bar(aes(x0 = x + 0.5, y0 = y + 0.5, # position the semicircles between the grid lines
                              r0 = 0, r = value_abs, # r = radius
                              start = start + pi, end = start, fill = value), # calculate fill color based on value
                          color = "white", linetype = "blank") +
    ggforce::geom_circle(data = data_lower, 
                aes(x0 = x + 0.5, y0 = y + 0.5, r = value_abs, fill = value), color = "white") + 
    ggforce::geom_circle(data = data_lower2, 
                aes(x0 = x + 0.5, y0 = y + 0.5, r = 0.05*cross0), color = "black", fill = "black") + 
    guides(r = guide_legend()) +
    xlab("") + ylab("") +
    labs(fill = "Strength of correlation\n(upper triangle) ") +
    scale_fill_gradientn(colors = gcolors, limits = limits) + # Limits = absolute limits of correlations (-1 and 1)
    scale_x_continuous(breaks = c(1.5:(n_corr + .5)), minor_breaks = c(1:(n_corr + 1)), 
                       labels = labels, position = "top",
                       expand = c(0,0)) +
    scale_y_continuous(breaks = c(-0.5:(n_corr*-1)), minor_breaks = c(0:(n_corr*-1)), 
                       labels = labels,
                       expand = c(0,0)) +
    expand_limits(x = c(1,(n_corr+1)), y = c(0,(n_corr*-1))) +
    coord_fixed() + theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 0), # angle of the text at the top
          panel.grid.major = element_blank(), # no major grid lines
          panel.grid.minor = element_line(color = "grey80"), # minor grid lines
          axis.ticks = element_blank(),
          axis.ticks.length = unit(0, "pt"), # remove axis ticks
          panel.border = element_rect(color = "grey80"), # border of the whole panel/plot
          plot.margin = unit(c(-0.4, 0, 0, -1.5), "cm"),
          text = element_text(size = 10),
          legend.position = "bottom")
  return(plot_twoclass)
}



#### Black and white corrplot with two classes, function doublecorrplot_bw ----
doublecorrplot_bw <- function(class1, class2, limits = c(-1,1), labels = rownames(class1)) {
  # Get 'coordinates' of correlations in matrix to be used as the position in the plot
  x <- as.vector(col(class1))
  y <- as.vector(row(class1)*-1)
  
  ### Upper triangle
  # Class 1 (on left = pi)
  class1_list <- as.vector(class1)
  class1_list <- cbind("value" = class1_list, "value_abs" = abs(class1_list), "start" = pi, x, y)
  
  # Class 2 (on right = 0)
  class2_list <- as.vector(class2)
  class2_list <- cbind("value" = class2_list, "value_abs" = abs(class2_list), "start" = 0, x, y)
  
  # Merge
  data_upper <- as.data.frame(rbind(class1_list, class2_list))
  data_upper <- data_upper[as.vector(upper.tri(class1)), ]  
  data_upper$value_abs <- scales:::rescale(data_upper$value_abs, c(0.15, 0.45)) # rescale
  
  ### Lower triangle
  diff <- as.vector(class1 - class2)
  data_lower <- as.data.frame(cbind(value = diff, value_abs = abs(diff), x, y))
  data_lower$value_abs <- scales:::rescale(abs(data_lower$value_abs), c(0.15,0.45), # rescale 
                                           from = range(abs(data_upper$value), na.rm = TRUE, finite = TRUE)) # to same scale as upper triangle
  data_lower <- data_lower[as.vector(lower.tri(class1)), ]  
  
  # Crosses 0?
  cross0 <- ifelse(as.vector(class1) > 0 & 0 > as.vector(class2) | as.vector(class2) > 0 & 0 > as.vector(class1), 1, NA_integer_)
  data_lower2 <- as.data.frame(cbind(cross0, x, y))
  data_lower2 <- data_lower2[as.vector(lower.tri(class1)), ]
  data_lower2 <- data_lower2[!is.na(data_lower2$cross0), ]  
  
  # Define colors (similar palette as corrplot)
  gcolors <- c("#67001F", "#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
               "#FFFFFF", 
               "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061")
  
  # Color limits set to max and min correlations?
  if (limits == "max") {
    limits <- rbind(grps$average1, grps$average2)
    limits <- ceiling(max(abs(limits), na.rm=T)*10)/10
    limits <- c(-limits, limits)
  }
  
  # Get number of correlations for the breaks in graph
  n_corr <- nrow(class1)
  
  # Plot the graph
  plot_twoclass <- ggplot(data_upper) + 
    ggforce::geom_arc_bar(aes(x0 = x + 0.5, y0 = y + 0.5, # position the semicircles between the grid lines
                              r0 = 0, r = value_abs, # r = radius
                              start = start + pi, end = start, fill = factor(sign(value), levels = c("1", "-1"))), # calculate fill color based on value
                          color = "white", size=0.05) +
    ggforce::geom_circle(data = data_lower, 
                aes(x0 = x + 0.5, y0 = y + 0.5, r = value_abs, fill = factor(sign(value), levels = c("1", "-1"))), color = "white") + 
    ggforce::geom_circle(data = data_lower2, 
                aes(x0 = x + 0.5, y0 = y + 0.5, r = 0.05*cross0), color = "white", fill = "white") + 
    guides(r = guide_legend()) +
    xlab("") + ylab("") +
    labs(fill = "Sign of correlation\n(upper triangle) ") +
    scale_fill_discrete(labels = c("Positive", "Negative"), type = c("gray27", "gray")) +
    scale_x_continuous(breaks = c(1.5:(n_corr + .5)), minor_breaks = c(1:(n_corr + 1)), 
                       labels = labels, position = "top",
                       expand = c(0,0)) +
    scale_y_continuous(breaks = c(-0.5:(n_corr*-1)), minor_breaks = c(0:(n_corr*-1)), 
                       labels = labels,
                       expand = c(0,0)) +
    expand_limits(x = c(1,(n_corr+1)), y = c(0,(n_corr*-1))) +
    coord_fixed() + theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 0), # angle of the text at the top
          panel.grid.major = element_blank(), # no major grid lines
          panel.grid.minor = element_line(color = "grey80"), # minor grid lines
          axis.ticks = element_blank(),
          axis.ticks.length = unit(0, "pt"), # remove axis ticks
          panel.border = element_rect(color = "grey80"), # border of the whole panel/plot
          plot.margin = unit(c(-0.4, 0, 0, -1.5), "cm"),
          text = element_text(size = 10),
          legend.position = "bottom")
  return(plot_twoclass)
}
