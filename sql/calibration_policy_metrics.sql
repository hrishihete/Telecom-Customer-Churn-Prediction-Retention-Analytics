-- Input: reports/calibration/evaluation_predictions.csv as evaluation_predictions.
-- One row per role; thresholds and flags were frozen using tuning outcomes only.
WITH counts AS (
    SELECT role, COUNT(*) AS customers,
           SUM(balanced_f1_flag) AS contacts,
           SUM(CASE WHEN balanced_f1_flag = 1 AND actual_churn = 1 THEN 1 ELSE 0 END) AS tp,
           SUM(CASE WHEN balanced_f1_flag = 1 AND actual_churn = 0 THEN 1 ELSE 0 END) AS fp,
           SUM(CASE WHEN balanced_f1_flag = 0 AND actual_churn = 1 THEN 1 ELSE 0 END) AS fn,
           SUM(CASE WHEN balanced_f1_flag = 0 AND actual_churn = 0 THEN 1 ELSE 0 END) AS tn
    FROM evaluation_predictions
    GROUP BY role
)
SELECT *, 1.0 * contacts / customers AS contact_rate,
       1.0 * tp / NULLIF(tp + fp, 0) AS precision,
       1.0 * tp / NULLIF(tp + fn, 0) AS recall,
       2.0 * tp / NULLIF(2 * tp + fp + fn, 0) AS f1
FROM counts
ORDER BY role;
