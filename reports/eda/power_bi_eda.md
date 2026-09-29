# Power BI EDA handoff

Import `data/processed/telco_churn_cleaned.csv` as `TelcoCustomers`, using whole numbers for tenure. This guide accompanies the Python EDA; it is not a completed PBIX file.

Create these measures:

```dax
Customers = COUNTROWS(TelcoCustomers)
Churned Customers = CALCULATE([Customers], TelcoCustomers[Churn] = "Yes")
Churn Rate = DIVIDE([Churned Customers], [Customers])
```

Format Churn Rate as a percentage. Create these calculated columns:

```dax
Tenure Band Order =
VAR t = TelcoCustomers[tenure]
RETURN SWITCH(TRUE(),
    ISBLANK(t) || t < 0 || t > 72, 99,
    t <= 12, 1, t <= 24, 2, t <= 36, 3,
    t <= 48, 4, t <= 60, 5, 6)

Tenure Band = SWITCH(TelcoCustomers[Tenure Band Order],
    1, "0–12 months", 2, "13–24 months", 3, "25–36 months",
    4, "37–48 months", 5, "49–60 months", 6, "61–72 months", "Unknown")
```

Sort Tenure Band by Tenure Band Order. Build three bar charts with Contract, PaymentMethod, and Tenure Band respectively on the category axis and Churn Rate as the value. Add Customers and Churned Customers to tooltips so small segments are visible. Add a matrix with Contract rows, Tenure Band columns, and Churn Rate values, with background color formatting.

With no filters applied, cards must show 7,043 customers, 1,869 churned customers, and 26.54% churn. Compare each visual's group counts and rates with the CSV outputs in this folder. Use measures on the customer-level data for interactive filtering; never average segment percentages. Avoid a Churn slicer, which can make a rate trivially 0% or 100%.
