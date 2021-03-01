# carparts-demo
  Labor costs are a significant portion of optimizable costs associated with inventory warehousing. Developing better labor hour allocation requires more accurate forecasting of client demand in conjunction with current employee resource counts. Herein, we demonstrate a method of segmenting historical

# Architecture
![architecture](https://github.com/brickmeister/carparts-demo/raw/main/images/Carparts%20Workshop.png)

# Libraries used
* [Spark DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html)
* [Delta Lake](https://docs.delta.io/latest/delta-intro.html)
* [SparkML](http://spark.apache.org/docs/latest/ml-guide.html)
* [MLFlow](https://www.mlflow.org/docs/latest/index.html)
* [SparkXGBoost](https://github.com/sllynn/spark-xgboost.git)

# Models used
* [Hierarchical Bisecting-K-Means](https://medium.com/@afrizalfir/bisecting-kmeans-clustering-5bc17603b8a2)
* [Decision Tree Classifier](https://medium.com/swlh/decision-tree-classification-de64fc4d5aac)
