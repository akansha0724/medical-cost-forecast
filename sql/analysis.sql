-- Medical Cost Forecasting — SQL Analysis
-- Source: CMS National Health Expenditure (NHE) Tables
-- Database: medical_costs | Table: nhe_spending

USE medical_costs;

-- ─────────────────────────────────────────────────────────────
-- Q1: Total spending by service line (all years combined)
-- Which service line has the highest cumulative spend?
-- ─────────────────────────────────────────────────────────────
SELECT
    service_line,
    label,
    ROUND(SUM(spending_bn), 1)  AS total_spend_bn,
    ROUND(AVG(spending_bn), 1)  AS avg_annual_spend_bn
FROM nhe_spending
GROUP BY service_line, label
ORDER BY total_spend_bn DESC;


-- ─────────────────────────────────────────────────────────────
-- Q2: Year-over-year growth rate per service line
-- Which lines are growing fastest?
-- ─────────────────────────────────────────────────────────────
SELECT
    a.year,
    a.service_line,
    a.spending_bn,
    b.spending_bn                                            AS prev_year_bn,
    ROUND((a.spending_bn - b.spending_bn) / b.spending_bn * 100, 2) AS yoy_growth_pct
FROM nhe_spending a
JOIN nhe_spending b
    ON a.service_line = b.service_line
    AND a.year = b.year + 1
ORDER BY a.service_line, a.year;


-- ─────────────────────────────────────────────────────────────
-- Q3: Most recent 10 years — spend trajectory per service line
-- Used to anchor the forecast period
-- ─────────────────────────────────────────────────────────────
SELECT
    year,
    service_line,
    spending_bn,
    ROUND(SUM(spending_bn) OVER (PARTITION BY service_line ORDER BY year
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) / 5, 1) AS rolling_5yr_avg
FROM nhe_spending
WHERE year >= (SELECT MAX(year) - 9 FROM nhe_spending)
ORDER BY service_line, year;


-- ─────────────────────────────────────────────────────────────
-- Q4: COVID impact — 2019 vs 2020 drop and 2021 recovery
-- Shows why event regressors matter in Prophet
-- ─────────────────────────────────────────────────────────────
SELECT
    service_line,
    MAX(CASE WHEN year = 2019 THEN spending_bn END) AS spend_2019,
    MAX(CASE WHEN year = 2020 THEN spending_bn END) AS spend_2020,
    MAX(CASE WHEN year = 2021 THEN spending_bn END) AS spend_2021,
    ROUND(
        (MAX(CASE WHEN year = 2020 THEN spending_bn END) -
         MAX(CASE WHEN year = 2019 THEN spending_bn END)) /
         MAX(CASE WHEN year = 2019 THEN spending_bn END) * 100, 2
    ) AS covid_impact_pct
FROM nhe_spending
WHERE year IN (2019, 2020, 2021)
GROUP BY service_line
ORDER BY covid_impact_pct;


-- ─────────────────────────────────────────────────────────────
-- Q5: Prescription drug spend vs hospital spend ratio over time
-- Tracks structural shift in where healthcare dollars flow
-- ─────────────────────────────────────────────────────────────
SELECT
    a.year,
    a.spending_bn                                           AS hospital_bn,
    b.spending_bn                                           AS rx_bn,
    ROUND(b.spending_bn / a.spending_bn * 100, 1)          AS rx_as_pct_of_hospital
FROM nhe_spending a
JOIN nhe_spending b ON a.year = b.year
WHERE a.service_line = 'hospital'
  AND b.service_line = 'prescription_drugs'
ORDER BY a.year;


-- ─────────────────────────────────────────────────────────────
-- Q6: Average growth rate by decade
-- Used to set long-run trend assumptions for forecast scenarios
-- ─────────────────────────────────────────────────────────────
SELECT
    service_line,
    CONCAT(FLOOR(year / 10) * 10, 's')                     AS decade,
    ROUND(AVG(spending_bn), 1)                              AS avg_spend_bn,
    COUNT(*)                                                AS years_in_decade
FROM nhe_spending
GROUP BY service_line, FLOOR(year / 10) * 10
ORDER BY service_line, decade;


-- ─────────────────────────────────────────────────────────────
-- Q7: Window function — rank service lines by spend each year
-- Shows which line dominates spending and when rankings shift
-- ─────────────────────────────────────────────────────────────
SELECT
    year,
    service_line,
    spending_bn,
    RANK() OVER (PARTITION BY year ORDER BY spending_bn DESC) AS spend_rank
FROM nhe_spending
WHERE year >= 2000
ORDER BY year, spend_rank;
