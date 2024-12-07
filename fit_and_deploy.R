# Libraries
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

# Access and prepare Snowflake data from R ----------

## Connect to database
con <- dbConnect(
  odbc::snowflake(),
  warehouse = "DEFAULT_WH",
  database = "LENDING_CLUB",
  schema = "ML"
)

## Load table and prepare for modeling
lendingclub_dat <-
  con |>
  tbl("LOAN_DATA") |>
  mutate(
    ISSUE_YEAR = as.integer(str_sub(ISSUE_D, start = 5)),
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
    )
  )

## Sample data
lendingclub_sample <-
  lendingclub_dat |>
  filter(ISSUE_YEAR == 2016) |>
  slice_sample(n = 5000)

## Select columns of interest for model
lendingclub_prep <-
  lendingclub_sample |>
  select(INT_RATE, TERM, BC_UTIL, BC_OPEN_TO_BUY, ALL_UTIL) |>
  mutate(INT_RATE = as.numeric(str_remove(INT_RATE, "%"))) |>
  filter(!if_any(everything(), is.na)) |>
  collect()

# Fit model with tidymodels ----------

## Create recipe for preprocessing
lendingclub_rec <-
  recipe(INT_RATE ~ ., data = lendingclub_prep) |>
  step_mutate(TERM = (TERM == "60 months")) |>
  step_mutate(across(!TERM, as.numeric)) |>
  step_normalize(all_numeric_predictors()) |>
  step_impute_mean(all_of(c("BC_OPEN_TO_BUY", "BC_UTIL"))) |>
  step_filter(!if_any(everything(), is.na))

## Create workflow
lendingclub_lr <- linear_reg()
lendingclub_wf <-
  workflow() |>
  add_model(lendingclub_lr) |>
  add_recipe(lendingclub_rec)

## Fit model
lendingclub_fit <-
  lendingclub_wf |>
  fit(data = lendingclub_prep)

## Compute model metrics
lendingclub_metric_set <- metric_set(rmse, mae, rsq)

lendingclub_metrics <-
  lendingclub_fit |>
  augment(lendingclub_prep) |>
  lendingclub_metric_set(truth = INT_RATE, estimate = .pred)

## Register model with vetiver
board <- board_connect()

v <-
  vetiver_model(
    lendingclub_fit,
    model_name,
    metadata = list(metrics = lendingclub_metrics)
  )
board |> vetiver_pin_write(v)

## Identify active model version
model_versions <-
  board |>
  pin_versions(glue("{board$account}/{model_name}"))

model_version <-
  model_versions |>
  filter(active) |>
  pull(version)

# Predict with orbital and Snowflake -------------

## Convert to orbital object
orbital_obj <- orbital(lendingclub_fit)

## Inspect model's full Snowflake SQL syntax
sql_predictor <- orbital_sql(orbital_obj, con)

start_time <- Sys.time()
## Run predictions directly in Snowflake, save results to temporary table
preds <-
  predict(orbital_obj, lendingclub_dat) |>
  compute(name = "LENDING_CLUB_PREDICTIONS_TEMP6")

end_time <- Sys.time()

# Count the number of predictions generated and calculate time spent
preds |> count()
end_time - start_time

# Deploy model as a Snowflake view ------------

## Write query for view
view_sql <-
  lendingclub_dat |>
  mutate(!!!orbital_inline(orbital_obj)) |>
  select(any_of(c("ID", ".pred"))) |>
  remote_query()

## Name for view
versioned_view_name <- glue("{model_name}_v{model_version}")

## Combine view_sql with SQL to create view
snowflake_view_statement <-
  glue::glue_sql(
    "CREATE OR REPLACE VIEW {`versioned_view_name`} AS ",
    view_sql,
    .con = con
  )

## Execute view creation query
con |>
  DBI::dbExecute(snowflake_view_statement)

## Create main view that stays in sync with latest model version
main_view_name <- glue::glue("{model_name}_latest")

main_view_statement <- glue::glue_sql(
  "CREATE OR REPLACE VIEW {`main_view_name`} AS ",
  "SELECT * FROM {`versioned_view_name`}",
  .con = con
)

con |>
  DBI::dbExecute(main_view_statement)

## Try out view
con |>
  tbl(main_view_name) |>
  head(100) |>
  collect()