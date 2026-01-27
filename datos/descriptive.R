library(tidyverse)
library(cluster)
library(janitor)

df_raw <- read_csv("balanced_dataset.csv") |>
  clean_names()

select_diverse_per_class <- function(df, y, n_per_class = 5) {
  # df: data.frame
  # y: nombre columna target (string)

  df %>%
    group_by(.data[[y]]) %>%
    group_modify(~{
      X <- .x %>%
        select(where(is.numeric)) %>%
        scale()

      # PAM devuelve medoids (observaciones reales)
      pam_fit <- pam(X, k = n_per_class)
      .x[pam_fit$id.med, , drop = FALSE]
    }) %>%
    ungroup()
}

n_per_classes <- 10

# uso
sampleN <- select_diverse_per_class(df_raw, y = "attrition", n_per_class = n_per_classes)

write_csv(sampleN, file = paste0(n_per_classes*2, "_cases_atttrition.csv"))