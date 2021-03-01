# Carparts Demo
  Labor costs are a significant portion of optimizable costs associated with inventory warehousing. Developing better labor hour allocation requires more accurate forecasting of client demand in conjunction with current employee resource counts. Herein, we demonstrate a method of segmenting historical

# Table of Contents
- [Architecture](#architecture)
- [Dataset](#dataset)
  - [Schema](#schema)
- [Libraries](#libraries-used)
- [Models](#models-used)

# Architecture
![architecture](https://github.com/brickmeister/carparts-demo/raw/main/images/Carparts%20Workshop.png)

# Dataset

Data used in this analysis was generously obtained from Carparts.

## Schema

| Field | Data Type | Description |
| ------------ | ------------ | ------------ |
| ""| IntegerType|Index ID |
| "Count Of Order Number"| IntegerType| Order Count |
| "Order Type"| StringType| Order Type|
| "WH ID"| StringType| Site Warehouse ID |
| "Date"| TimestampType| Date of request|å
| "Year"| IntegerType| Year of request|
| "Week_Number"| IntegerType| Week Number of the Year |
| "Days_Until_IRS_Refund"| IntegerType| Days Until IRS Refund |
| "Days_Until_Stimulus_Check"| IntegerType| Days Until USA Federal Government Stimulus Check |

# Libraries used
* [Spark DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html)
* [Delta Lake](https://docs.delta.io/latest/delta-intro.html)
* [SparkML](http://spark.apache.org/docs/latest/ml-guide.html)
* [MLFlow](https://www.mlflow.org/docs/latest/index.html)
* [SparkXGBoost](https://github.com/sllynn/spark-xgboost.git)

# Models used
* [Hierarchical Bisecting-K-Means](https://medium.com/@afrizalfir/bisecting-kmeans-clustering-5bc17603b8a2)
* [Decision Tree Classifier](https://medium.com/swlh/decision-tree-classification-de64fc4d5aac)
