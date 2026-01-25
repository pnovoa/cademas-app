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

# uso
sample10 <- select_diverse_per_class(df_raw, y = "attrition", n_per_class = 5)

write_csv(sample10, file = "10_cases_atttrition.csv")