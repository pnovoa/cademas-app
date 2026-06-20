#!/usr/bin/env Rscript

# Build a compact, human-readable feature summary table for the attrition example.
# The script intentionally uses base R + grid so the PDF can be rendered without
# browser/HTML dependencies such as gt + webshot2.

suppressPackageStartupMessages({
  library(grid)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else "paper/scripts/make_attrition_feature_summary_table.R"
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = FALSE)
if (!dir.exists(file.path(repo_root, "example_attrition"))) {
  repo_root <- normalizePath(getwd(), mustWork = TRUE)
}

data_path <- file.path(repo_root, "example_attrition", "data", "cases_atttrition.csv")
out_dir <- file.path(repo_root, "paper", "tables")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_pdf_env <- Sys.getenv("OUT_PDF", unset = "")
out_pdf <- if (nzchar(out_pdf_env)) normalizePath(out_pdf_env, mustWork = FALSE) else file.path(out_dir, "attrition_feature_summary_table.pdf")

raw <- read.csv(data_path, sep = ";", check.names = FALSE, stringsAsFactors = FALSE)
feature_names <- setdiff(names(raw), "Case_ID")

target <- factor(raw$attrition, levels = c("No", "Yes"))
attrition_cols <- c(No = "#c8d1d8", Yes = "#e13f40")
grid_col <- "#d5dadd"
text_col <- "#202020"
muted_col <- "#666666"

feature_labels <- c(
  attrition = "Attrition (target)",
  age = "Age",
  business_travel = "Business travel",
  daily_rate = "Daily rate",
  department = "Department",
  distance_from_home = "Distance from home",
  education = "Education",
  education_field = "Education field",
  employee_count = "Employee count",
  employee_number = "Employee number",
  environment_satisfaction = "Environment satisfaction",
  gender = "Gender",
  hourly_rate = "Hourly rate",
  job_involvement = "Job involvement",
  job_level = "Job level",
  job_role = "Job role",
  job_satisfaction = "Job satisfaction",
  marital_status = "Marital status",
  monthly_income = "Monthly income",
  monthly_rate = "Monthly rate",
  num_companies_worked = "Number of companies worked",
  over18 = "Over 18",
  over_time = "Overtime",
  percent_salary_hike = "Percent salary hike",
  performance_rating = "Performance rating",
  relationship_satisfaction = "Relationship satisfaction",
  standard_hours = "Standard hours",
  stock_option_level = "Stock option level",
  total_working_years = "Total working years",
  training_times_last_year = "Training times last year",
  work_life_balance = "Work-life balance",
  years_at_company = "Years at company",
  years_in_current_role = "Years in current role",
  years_since_last_promotion = "Years since last promotion",
  years_with_curr_manager = "Years with current manager"
)

categorical_features <- c(
  "attrition", "business_travel", "department", "education_field", "gender",
  "job_role", "marital_status", "over18", "over_time"
)

ordinal_features <- c(
  "education", "environment_satisfaction", "job_involvement", "job_level",
  "job_satisfaction", "performance_rating", "relationship_satisfaction",
  "stock_option_level", "work_life_balance"
)

ordinal_level_labels <- list(
  education = c("1" = "Below college", "2" = "College", "3" = "Bachelor", "4" = "Master", "5" = "Doctor"),
  environment_satisfaction = c("1" = "Low", "2" = "Medium", "3" = "High", "4" = "Very high"),
  job_involvement = c("1" = "Low", "2" = "Medium", "3" = "High", "4" = "Very high"),
  job_satisfaction = c("1" = "Low", "2" = "Medium", "3" = "High", "4" = "Very high"),
  relationship_satisfaction = c("1" = "Low", "2" = "Medium", "3" = "High", "4" = "Very high"),
  work_life_balance = c("1" = "Bad", "2" = "Good", "3" = "Better", "4" = "Best"),
  performance_rating = c("1" = "Low", "2" = "Good", "3" = "Excellent", "4" = "Outstanding"),
  stock_option_level = c("0" = "None", "1" = "Low", "2" = "Medium", "3" = "High"),
  job_level = c("1" = "Level 1", "2" = "Level 2", "3" = "Level 3", "4" = "Level 4", "5" = "Level 5")
)

humanize_value <- function(x) {
  x <- as.character(x)
  x <- gsub("_", " ", x, fixed = TRUE)
  x <- gsub("([a-z])([A-Z])", "\\1 \\2", x)
  x <- ifelse(x == "Y", "Yes", x)
  x <- ifelse(x == "N", "No", x)
  x <- gsub("Non Travel", "Non-travel", x)
  x <- gsub("Travel Rarely", "Travel rarely", x)
  x <- gsub("Travel Frequently", "Travel frequently", x)
  x <- gsub("Technical Degree", "Technical degree", x)
  x
}

feature_type <- function(name, values) {
  if (name == "attrition") return("Target")
  if (name == "employee_number") return("Identifier-like")
  if (name %in% categorical_features) return("Categorical")
  if (name %in% ordinal_features) return("Ordinal")
  if (length(unique(values[!is.na(values)])) <= 1) return("Constant")
  "Numeric"
}

fmt_num <- function(x, digits = 1) {
  ifelse(abs(x) >= 1000, format(round(x, digits), big.mark = ",", trim = TRUE), format(round(x, digits), trim = TRUE))
}

summarize_numeric <- function(x) {
  x <- as.numeric(x)
  qs <- quantile(x, c(0.25, 0.5, 0.75), na.rm = TRUE, names = FALSE)
  sprintf(
    "Mean %s (SD %s); median %s [%s, %s]",
    fmt_num(mean(x, na.rm = TRUE)),
    fmt_num(sd(x, na.rm = TRUE)),
    fmt_num(qs[2]),
    fmt_num(qs[1]),
    fmt_num(qs[3])
  )
}

summarize_identifier <- function(x) {
  n_unique <- length(unique(x[!is.na(x)]))
  sprintf("%d unique values; retained in CSV but not used by the example models", n_unique)
}

summarize_categorical <- function(x, name = NULL, max_items = 4) {
  if (!is.null(name) && name %in% names(ordinal_level_labels)) {
    mapped <- ordinal_level_labels[[name]][as.character(x)]
    x <- ifelse(is.na(mapped), as.character(x), mapped)
  } else {
    x <- humanize_value(x)
  }
  tab <- sort(table(x), decreasing = TRUE)
  n <- sum(tab)
  items <- sprintf("%s %d (%d%%)", names(tab), as.integer(tab), round(100 * as.integer(tab) / n))
  if (length(items) > max_items) {
    items <- c(items[seq_len(max_items)], sprintf("+%d more", length(items) - max_items))
  }
  paste(items, collapse = "; ")
}

levels_or_range <- function(name, x, type) {
  if (type == "Identifier-like") {
    sprintf("Numeric identifier; range %s-%s", fmt_num(min(x, na.rm = TRUE)), fmt_num(max(x, na.rm = TRUE)))
  } else if (type %in% c("Numeric", "Constant")) {
    ux <- unique(x[!is.na(x)])
    sprintf("Range %s-%s; %d observed value%s", fmt_num(min(x, na.rm = TRUE)), fmt_num(max(x, na.rm = TRUE)), length(ux), ifelse(length(ux) == 1, "", "s"))
  } else if (type == "Ordinal" && name %in% names(ordinal_level_labels)) {
    observed <- sort(unique(as.character(x)))
    labs <- ordinal_level_labels[[name]][observed]
    paste(sprintf("%s = %s", observed, labs), collapse = "; ")
  } else {
    summarize_categorical(x, name, max_items = 8)
  }
}

wrap_label <- function(x, width) {
  paste(strwrap(x, width = width), collapse = "\n")
}

draw_text <- function(label, x, y, w, h, size = 6.5, fontface = "plain", col = text_col) {
  grid.text(
    wrap_label(label, max(8, floor(w * 115))),
    x = unit(x + 0.006, "npc"),
    y = unit(y + h / 2, "npc"),
    just = c("left", "center"),
    gp = gpar(fontsize = size, fontface = fontface, col = col, lineheight = 0.92)
  )
}

draw_bar_rect <- function(x0, y0, w, h, fill, col = "white") {
  grid.rect(x = unit(x0 + w / 2, "npc"), y = unit(y0 + h / 2, "npc"),
            width = unit(w, "npc"), height = unit(h, "npc"),
            gp = gpar(fill = fill, col = col, lwd = 0.25))
}

draw_overall_plot <- function(values, type, name, x0, y0, w, h) {
  pad_x <- w * 0.08
  pad_y <- h * 0.18
  plot_x <- x0 + pad_x
  plot_y <- y0 + pad_y
  plot_w <- w - 2 * pad_x
  plot_h <- h - 2 * pad_y
  grid.rect(unit(plot_x + plot_w / 2, "npc"), unit(plot_y + plot_h / 2, "npc"),
            unit(plot_w, "npc"), unit(plot_h, "npc"), gp = gpar(fill = "#f7f7f7", col = "#eeeeee", lwd = 0.2))
  if (type == "Identifier-like") {
    grid.text("identifier", x = unit(plot_x + plot_w / 2, "npc"), y = unit(plot_y + plot_h / 2, "npc"),
              gp = gpar(fontsize = 5.5, col = muted_col))
  } else if (type %in% c("Numeric", "Constant")) {
    z <- as.numeric(values)
    if (length(unique(z)) <= 1) {
      grid.lines(unit(c(plot_x, plot_x + plot_w), "npc"), unit(c(plot_y + plot_h / 2, plot_y + plot_h / 2), "npc"),
                 gp = gpar(col = "#9aa4aa", lwd = 0.7))
      grid.points(unit(plot_x + plot_w / 2, "npc"), unit(plot_y + plot_h / 2, "npc"), pch = 16, size = unit(1.5, "mm"), gp = gpar(col = "#59656d"))
    } else {
      br <- pretty(range(z, na.rm = TRUE), n = 6)
      counts <- hist(z, breaks = br, plot = FALSE)$counts
      if (sum(counts) == 0) return(invisible(NULL))
      bw <- plot_w / length(counts)
      for (i in seq_along(counts)) {
        bh <- plot_h * counts[i] / max(counts)
        draw_bar_rect(plot_x + (i - 1) * bw + bw * 0.08, plot_y, bw * 0.84, bh, "#9fb7c9", "#50616f")
      }
    }
  } else {
    labs <- if (type == "Ordinal" && name %in% names(ordinal_level_labels)) {
      mapped <- ordinal_level_labels[[name]][as.character(values)]
      ifelse(is.na(mapped), as.character(values), mapped)
    } else {
      humanize_value(values)
    }
    tab <- table(labs)
    counts <- as.numeric(tab)
    bw <- plot_w / length(counts)
    for (i in seq_along(counts)) {
      bh <- plot_h * counts[i] / max(counts)
      draw_bar_rect(plot_x + (i - 1) * bw + bw * 0.08, plot_y, bw * 0.84, bh, "#b9c3ca", "#46545c")
    }
  }
}

draw_by_attrition_plot <- function(values, type, name, x0, y0, w, h) {
  if (name == "attrition") {
    draw_overall_plot(values, "Target", name, x0, y0, w, h)
    return(invisible(NULL))
  }
  pad_x <- w * 0.08
  pad_y <- h * 0.18
  plot_x <- x0 + pad_x
  plot_y <- y0 + pad_y
  plot_w <- w - 2 * pad_x
  plot_h <- h - 2 * pad_y
  grid.rect(unit(plot_x + plot_w / 2, "npc"), unit(plot_y + plot_h / 2, "npc"),
            unit(plot_w, "npc"), unit(plot_h, "npc"), gp = gpar(fill = "#f7f7f7", col = "#eeeeee", lwd = 0.2))

  if (type == "Identifier-like") {
    grid.text("not summarized", x = unit(plot_x + plot_w / 2, "npc"), y = unit(plot_y + plot_h / 2, "npc"),
              gp = gpar(fontsize = 5.5, col = muted_col))
  } else if (type %in% c("Numeric", "Constant")) {
    z <- as.numeric(values)
    rng <- range(z, na.rm = TRUE)
    if (diff(rng) == 0) rng <- rng + c(-0.5, 0.5)
    y_levels <- c(No = plot_y + plot_h * 0.68, Yes = plot_y + plot_h * 0.32)
    for (lev in names(y_levels)) {
      zz <- z[target == lev]
      if (!length(zz)) next
      qs <- quantile(zz, c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE, names = FALSE)
      xs <- plot_x + plot_w * (qs - rng[1]) / diff(rng)
      yy <- y_levels[[lev]]
      grid.lines(unit(c(xs[1], xs[5]), "npc"), unit(c(yy, yy), "npc"), gp = gpar(col = "#333333", lwd = 0.45))
      grid.rect(unit((xs[2] + xs[4]) / 2, "npc"), unit(yy, "npc"), unit(max(xs[4] - xs[2], 0.002), "npc"), unit(plot_h * 0.22, "npc"),
                gp = gpar(fill = attrition_cols[[lev]], col = "#333333", lwd = 0.35))
      grid.lines(unit(c(xs[3], xs[3]), "npc"), unit(c(yy - plot_h * 0.12, yy + plot_h * 0.12), "npc"), gp = gpar(col = "#111111", lwd = 0.45))
    }
  } else {
    labs <- if (type == "Ordinal" && name %in% names(ordinal_level_labels)) {
      mapped <- ordinal_level_labels[[name]][as.character(values)]
      ifelse(is.na(mapped), as.character(values), mapped)
    } else {
      humanize_value(values)
    }
    all_labs <- unique(labs)
    all_labs <- all_labs[order(all_labs)]
    ncat <- length(all_labs)
    max_count <- max(table(labs, target))
    if (max_count == 0) max_count <- 1
    bw <- plot_w / max(ncat, 1)
    for (i in seq_along(all_labs)) {
      for (lev in c("No", "Yes")) {
        count <- sum(labs == all_labs[i] & target == lev)
        bar_w <- bw * 0.35
        offset <- ifelse(lev == "No", bw * 0.12, bw * 0.52)
        bh <- plot_h * count / max_count
        draw_bar_rect(plot_x + (i - 1) * bw + offset, plot_y, bar_w, bh, attrition_cols[[lev]], "#333333")
      }
    }
  }
}

rows <- lapply(feature_names, function(nm) {
  values <- raw[[nm]]
  type <- feature_type(nm, values)
  list(
    name = nm,
    label = feature_labels[[nm]] %||% humanize_value(nm),
    type = type,
    levels = levels_or_range(nm, values, type),
    summary = if (type == "Identifier-like") {
      summarize_identifier(values)
    } else if (type %in% c("Numeric", "Constant")) {
      summarize_numeric(values)
    } else {
      summarize_categorical(values, nm, max_items = 5)
    },
    values = values
  )
})

pdf(out_pdf, width = 11.7, height = 8.3, onefile = TRUE, family = "Helvetica")
on.exit(dev.off(), add = TRUE)

cols <- c(0.025, 0.175, 0.265, 0.485, 0.690, 0.835, 0.975)
headers <- c("Feature", "Type", "Levels / range", "Main summary", "Overall", "By attrition")
rows_per_page <- 12
num_pages <- ceiling(length(rows) / rows_per_page)

for (page in seq_len(num_pages)) {
  grid.newpage()
  grid.rect(gp = gpar(fill = "white", col = NA))
  grid.text("Attrition Example: Feature Summary Table",
            x = unit(0.025, "npc"), y = unit(0.955, "npc"),
            just = c("left", "center"), gp = gpar(fontsize = 13, fontface = "bold", col = text_col))
  grid.text("35 dataset columns excluding Case_ID; plots show overall distributions and distributions by attrition target.",
            x = unit(0.025, "npc"), y = unit(0.925, "npc"),
            just = c("left", "center"), gp = gpar(fontsize = 7.5, col = muted_col))
  grid.text(sprintf("Page %d/%d", page, num_pages),
            x = unit(0.975, "npc"), y = unit(0.955, "npc"),
            just = c("right", "center"), gp = gpar(fontsize = 7.5, col = muted_col))

  y_top <- 0.885
  header_h <- 0.045
  row_h <- 0.064
  grid.rect(unit(0.5, "npc"), unit(y_top - header_h / 2, "npc"),
            unit(0.95, "npc"), unit(header_h, "npc"),
            gp = gpar(fill = "#eef2f4", col = grid_col, lwd = 0.6))
  for (i in seq_along(headers)) {
    draw_text(headers[i], cols[i], y_top - header_h, cols[i + 1] - cols[i], header_h, size = 7.2, fontface = "bold")
  }
  for (x in cols) {
    grid.lines(unit(c(x, x), "npc"), unit(c(0.10, y_top), "npc"), gp = gpar(col = grid_col, lwd = 0.4))
  }

  idx <- ((page - 1) * rows_per_page + 1):min(page * rows_per_page, length(rows))
  for (j in seq_along(idx)) {
    r <- rows[[idx[j]]]
    y1 <- y_top - header_h - (j - 1) * row_h
    y0 <- y1 - row_h
    fill <- ifelse(j %% 2 == 1, "white", "#fbfcfd")
    grid.rect(unit(0.5, "npc"), unit(y0 + row_h / 2, "npc"),
              unit(0.95, "npc"), unit(row_h, "npc"),
              gp = gpar(fill = fill, col = grid_col, lwd = 0.35))
    draw_text(r$label, cols[1], y0, cols[2] - cols[1], row_h, size = 6.7, fontface = ifelse(r$name == "attrition", "bold", "plain"))
    draw_text(r$type, cols[2], y0, cols[3] - cols[2], row_h, size = 6.2, col = ifelse(r$type == "Target", "#7a1d1d", text_col))
    draw_text(r$levels, cols[3], y0, cols[4] - cols[3], row_h, size = 5.5, col = muted_col)
    draw_text(r$summary, cols[4], y0, cols[5] - cols[4], row_h, size = 5.7)
    draw_overall_plot(r$values, r$type, r$name, cols[5], y0, cols[6] - cols[5], row_h)
    draw_by_attrition_plot(r$values, r$type, r$name, cols[6], y0, cols[7] - cols[6], row_h)
  }

  legend_y <- 0.055
  grid.text("By attrition legend:", x = unit(0.025, "npc"), y = unit(legend_y, "npc"),
            just = c("left", "center"), gp = gpar(fontsize = 7, col = muted_col))
  draw_bar_rect(0.130, legend_y - 0.008, 0.014, 0.016, attrition_cols[["No"]], "#333333")
  grid.text("No", x = unit(0.150, "npc"), y = unit(legend_y, "npc"), just = c("left", "center"), gp = gpar(fontsize = 7, col = muted_col))
  draw_bar_rect(0.180, legend_y - 0.008, 0.014, 0.016, attrition_cols[["Yes"]], "#333333")
  grid.text("Yes", x = unit(0.200, "npc"), y = unit(legend_y, "npc"), just = c("left", "center"), gp = gpar(fontsize = 7, col = muted_col))
  grid.text("Numeric rows use histograms and attrition-specific mini boxplots; categorical/ordinal rows use count bars.",
            x = unit(0.975, "npc"), y = unit(legend_y, "npc"),
            just = c("right", "center"), gp = gpar(fontsize = 7, col = muted_col))
}

message("Wrote: ", out_pdf)
