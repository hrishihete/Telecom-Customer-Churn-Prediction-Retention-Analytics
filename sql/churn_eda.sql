-- Import telco_churn_cleaned.csv as telco_customers. Verified with SQLite.
SELECT Contract AS segment, COUNT(*) AS customers,
 SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
 100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) AS churn_rate_pct
FROM telco_customers GROUP BY Contract;

SELECT PaymentMethod AS segment, COUNT(*) AS customers,
 SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
 100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) AS churn_rate_pct
FROM telco_customers GROUP BY PaymentMethod;

WITH banded AS (
 SELECT Churn, CASE
 WHEN tenure BETWEEN 0 AND 12 THEN '0–12 months'
 WHEN tenure BETWEEN 13 AND 24 THEN '13–24 months'
 WHEN tenure BETWEEN 25 AND 36 THEN '25–36 months'
 WHEN tenure BETWEEN 37 AND 48 THEN '37–48 months'
 WHEN tenure BETWEEN 49 AND 60 THEN '49–60 months'
 WHEN tenure BETWEEN 61 AND 72 THEN '61–72 months'
 ELSE 'Unknown' END AS segment
 FROM telco_customers
)
SELECT segment, COUNT(*) AS customers,
 SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
 100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) AS churn_rate_pct
FROM banded GROUP BY segment;
