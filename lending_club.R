library(odbc)
library(DBI)
library(dbplyr)
library(dplyr)
library(glue)
library(arrow)
library(stringr)
library(tidymodels)
library(vetiver)
library(rsconnect)
library(pins)

con <- dbConnect(
  odbc::snowflake(),
  warehouse = "DEFAULT_WH",
  database = "LENDING_CLUB",
  schema = "ML"
)

model_name <- "interest_rate_prediction"

lendingclub_dat <-
  con |>
  tbl("LOAN_DATA") |>
  mutate(ISSUE_YEAR = as.integer(str_sub(ISSUE_D, start = 5)),
         ISSUE_MONTH = as.integer(
           case_match(
             str_sub(ISSUE_D, end = 3),
             "Jan" ~ 1,
             "Feb" ~ 2,
             "Mar" ~ 3,
             "Apr" ~ 4,
             "May" ~ 5,
             "Jun" ~ 6,
             "Jul" ~ 7,
             "Aug" ~ 8,
             "Sep" ~ 9,
             "Oct" ~ 10,
             "Nov" ~ 11,
             "Dec" ~ 12
           )
         ))

lendingclub_sample <-
  lendingclub_dat |>
  filter(ISSUE_YEAR == 2016) |>
  slice_sample(n = 5000)

# Select columns of interest for model
# Collect data
lendingclub_prep <-
  lendingclub_sample |>
  select(INT_RATE, TERM, BC_UTIL, BC_OPEN_TO_BUY, ALL_UTIL) |>
  mutate(INT_RATE = as.numeric(str_remove(INT_RATE, "%"))) |>
  filter(!if_any(everything(), is.na)) |>
  collect()

# Preprocessing
lendingclub_rec <-
  recipe(INT_RATE ~ ., data = lendingclub_prep) |>
  step_mutate(TERM = (TERM == "60 months")) |>
  step_mutate(across(!TERM, as.numeric)) |>
  step_normalize(all_numeric_predictors()) |>
  step_impute_mean(all_of(c("BC_OPEN_TO_BUY", "BC_UTIL"))) |>
  step_filter(!if_any(everything(), is.na))

# Create tidymodels workflow
lendingclub_lr <- linear_reg()

lendingclub_wf <-
  workflow() |>
  add_model(lendingclub_lr) |>
  add_recipe(lendingclub_rec)

# Fit model
lendingclub_fit <-
  lendingclub_wf |>
  fit(data = lendingclub_prep)

# Compute model metrics
lendingclub_metric_set <- metric_set(rmse, mae, rsq)

lendingclub_metrics <-
  lendingclub_fit |>
  augment(lendingclub_prep) |>
  lendingclub_metric_set(truth = INT_RATE, estimate = .pred)

# Register model
board <- board_connect()

v <- vetiver_model(lendingclub_fit,
                   model_name,
                   metadata = list(metrics = lendingclub_metrics))

board |> vetiver_pin_write(v)
