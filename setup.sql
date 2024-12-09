CREATE DATABASE IF NOT EXISTS LENDING_CLUB;
GRANT USAGE ON DATABASE LENDING_CLUB TO ROLE SYSADMIN;
GRANT USAGE ON FUTURE SCHEMAS IN DATABASE LENDING_CLUB TO ROLE SYSADMIN;
USE DATABASE LENDING_CLUB;

CREATE SCHEMA IF NOT EXISTS ML;
GRANT ALL PRIVILEGES ON FUTURE TABLES IN SCHEMA ML TO ROLE SYSADMIN;
USE SCHEMA ML;

CREATE STAGE LOAN_DATA
    URL = 's3://posit-snowflake-mlops';

CREATE OR REPLACE FILE FORMAT FORMAT_LENDING_CLUB_PARQUET
    TYPE = PARQUET
    NULL_IF = ('NULL', 'null');

CREATE OR REPLACE TABLE LOAN_DATA
    USING TEMPLATE (
        SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
        FROM TABLE(
            INFER_SCHEMA(
                LOCATION => '@LENDING_CLUB.ML.LOAN_DATA/loan_data.parquet',
                FILE_FORMAT => 'LENDING_CLUB.ML.FORMAT_LENDING_CLUB_PARQUET'
            )
        )
    );

COPY INTO LOAN_DATA
    FROM '@LENDING_CLUB.ML.LOAN_DATA/loan_data.parquet'
    FILE_FORMAT = (FORMAT_NAME = LENDING_CLUB.ML.FORMAT_LENDING_CLUB_PARQUET)
    MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;