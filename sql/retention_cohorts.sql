-- SQLite/standard aggregate syntax. Import the enriched active customer table
-- produced in scripts/retention_strategy.py as customer_risk.
-- Cohort combines Contract, tenure band, and InternetService (subscription proxy).
SELECT Cohort,
       COUNT(*) AS HighRiskCustomers,
       SUM(MonthlyCharges) AS MonthlyChargesCU,
       SUM(MonthlyCharges * RiskScore) AS WeightedExposureCU
FROM customer_risk
WHERE RiskScore >= 0.50
GROUP BY Cohort
ORDER BY WeightedExposureCU DESC;
