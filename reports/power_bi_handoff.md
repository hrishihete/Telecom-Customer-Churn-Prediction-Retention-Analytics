# Power BI Desktop handoff

This is an import guide for the completed inspection task; a `.pbix` dashboard is not included.

1. In Power BI Desktop, choose **Get data > Text/CSV**, select `data/processed/telco_churn_typed.csv`, and choose **Transform Data**.
2. Name the query `TelcoCustomers`. Use text for customerID, Churn, and all service/contract/payment categories. Use whole numbers for tenure and SeniorCitizen. Use decimal numbers for MonthlyCharges and TotalCharges, with an English (United States) locale if your default locale does not use decimal points.
3. Ensure the 11 empty TotalCharges cells are null, not zero. Keep all 7,043 rows. SeniorCitizen is a 0/1 category even though its storage type is numeric.
4. Choose **Close & Apply**. Add the following measures.

```dax
Customers = COUNTROWS(TelcoCustomers)

Churned Customers =
CALCULATE([Customers], TelcoCustomers[Churn] = "Yes")

Churn Rate = DIVIDE([Churned Customers], [Customers])

Missing Total Charges =
COUNTROWS(FILTER(TelcoCustomers, ISBLANK(TelcoCustomers[TotalCharges])))
```

Format Churn Rate as a percentage with two decimal places. With no filters applied, cards should show 7,043 customers, 1,869 churned customers, 26.54% churn, and 11 missing TotalCharges. A column chart of Churn by Customers should show No = 5,174 and Yes = 1,869. Contract and InternetService can be added as slicers for exploration; measures then describe the selected segment.

Do not label these measures as future churn risk or monthly churn trends: this file has historical target labels but no prediction scores or observation dates. It also lacks direct complaint records and detailed consumption data.
