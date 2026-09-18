-- Import data/processed/telco_churn_typed.csv as telco_customers first.
-- Blank TotalCharges must be SQL NULL; charges must be numeric.
-- Queries verified with SQLite; ordinary ANSI constructs are used.

-- Dataset size and identifier completeness.
SELECT COUNT(*) AS customers,
       COUNT(DISTINCT customerID) AS distinct_customers,
       SUM(CASE WHEN customerID IS NULL THEN 1 ELSE 0 END) AS missing_ids
FROM telco_customers;

-- Target imbalance; percentages use the entire dataset as denominator.
SELECT Churn, COUNT(*) AS customers,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM telco_customers), 2) AS percent
FROM telco_customers
GROUP BY Churn
ORDER BY Churn;

-- Missing charges and their relationship to zero tenure.
SELECT SUM(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END) AS missing_total_charges,
       SUM(CASE WHEN TotalCharges IS NULL AND tenure = 0 THEN 1 ELSE 0 END)
           AS missing_charges_with_zero_tenure
FROM telco_customers;

-- Duplicate identifier audit: expected to return no rows.
SELECT customerID, COUNT(*) AS occurrences
FROM telco_customers
GROUP BY customerID
HAVING COUNT(*) > 1;
