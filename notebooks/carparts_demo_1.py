# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Abstract
# MAGIC   Labor costs are a significant portion of optimizable costs associated with inventory warehousing. Developing better labor hour allocation requires more accurate forecasting of client demand in conjunction with current employee resource counts. Herein, we demonstrate a method of segmenting historical

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Dataset
# MAGIC 
# MAGIC Data used in this analysis was generously obtained from Carparts and displayed below in the following table
# MAGIC 
# MAGIC |COUNT OF ORDER NUMBER	|DATE	|ORDER TYPE|	WH ID|	Date	|Year	|Week_Number	|Days_Until_IRS_Refund|	Days_Until_Stimulus_Check|
# MAGIC |-|-|-|-|-|-|-|-|-|-|
# MAGIC |23|	1997|	1/2/19|	SHELFRPK|	LSL|	1/2/19|	2019|	1|	40|	467|
# MAGIC |51|	2001|	1/3/19|	SHELFRPK|	LSL|	1/3/19|	2019|	1|	39|	466|
# MAGIC |80|	1999|	1/4/19|	SHELFRPK|	LSL|	1/4/19|	2019|	1|	38|	465|
# MAGIC |106|	1689|	1/5/19|	SHELFRPK|	LSL|	1/5/19|	2019|	1|	37|	464|
# MAGIC |131|	1602|	1/6/19|	SHELFRPK|	LSL|	1/6/19|	2019|	2|	36|	463|
# MAGIC |158|	2235|	1/7/19|	SHELFRPK|	LSL|	1/7/19|	2019|	2|	35|	462|
# MAGIC |186|	2028|	1/8/19|	SHELFRPK|	LSL|	1/8/19|	2019|	2|	34|	461|
# MAGIC |214|	1766|	1/9/19|	SHELFRPK|	LSL|	1/9/19|	2019|	2|	33|	460|
# MAGIC |242|	1771|	1/10/19|	SHELFRPK|	LSL|	1/10/19|	2019|	2|	32|	459|
# MAGIC |270|	1796|	1/11/19|	SHELFRPK|	LSL|	1/11/19|	2019|	2|	31|	458|

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Schema
# MAGIC 
# MAGIC | Field | Data Type | Description |
# MAGIC | ------------ | ------------ | ------------ |
# MAGIC | ""| IntegerType|Index ID |
# MAGIC | "Count Of Order Number"| IntegerType| Order Count |
# MAGIC | "Order Type"| StringType| Order Type|
# MAGIC | "WH ID"| StringType| Site Warehouse ID |
# MAGIC | "Date"| TimestampType| Date of request|
# MAGIC | "Year"| IntegerType| Year of request|
# MAGIC | "Week_Number"| IntegerType| Week Number of the Year |
# MAGIC | "Days_Until_IRS_Refund"| IntegerType| Days Until IRS Refund |
# MAGIC | "Days_Until_Stimulus_Check"| IntegerType| Days Until USA Federal Government Stimulus Check |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Methods
# MAGIC 
# MAGIC Data is first cleaned and featurized using Scala SparkML libraries. Resultant data is passed through two machine learning models (hierarchical bisecting-kmeans, decision tree classifier) in order to segment a dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Architecture
# MAGIC <img src="https://raw.githubusercontent.com/bluerider/jhu_population_segmentation/main/img/JHU%20Demographic%20Segmentation.png" width = 100% /img>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ### Libraries Used
# MAGIC * [Spark DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html)
# MAGIC * [Delta Lake](https://docs.delta.io/latest/delta-intro.html)
# MAGIC * [SparkML](http://spark.apache.org/docs/latest/ml-guide.html)
# MAGIC * [MLFlow](https://www.mlflow.org/docs/latest/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Models Used
# MAGIC * [Hierarchical Bisecting-K-Means](https://medium.com/@afrizalfir/bisecting-kmeans-clustering-5bc17603b8a2)
# MAGIC * [Decision Tree Classifier](https://medium.com/swlh/decision-tree-classification-de64fc4d5aac)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Conclusion/Results
# MAGIC Six categories were derived from the unsupervised clustering which were classified according to a decision tree. Average accuracy for the decision tree model was 97% [CMD 77], and the average precision was 94% [CMD 79].
# MAGIC 
# MAGIC Among the features that strongly divided groups, the top features
# MAGIC were unsurprisingly # of patients recovered, # of active cases. The surprising factor was that the longitude of a record was also a strong separator into categories. This suggests that COVID-19 may have more stratification across timezones than seasons (heat doesn't strongly affect the virus).
# MAGIC 
# MAGIC Obviously due to the preliminary nature of this analysis, any conclusions drawn from this analysis should be further verified.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Follow up sources
# MAGIC 1. [Six distinct types of COVID-19](https://www.kcl.ac.uk/news/six-distinct-types-of-covid-19-identified)
# MAGIC 2. [Do weather conditions influence the transmission fo the coronavirus (SARS-CoV-2)](https://www.cebm.net/covid-19/do-weather-conditions-influence-the-transmission-of-the-coronavirus-sars-cov-2/)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Retrieve John's Hopkins Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve Data from Github

# COMMAND ----------

# DBTITLE 1,Download Data From GIT
# MAGIC %sh
# MAGIC 
# MAGIC ## get COVID-19 data from JHU CSSE
# MAGIC ## SOURCE : 
# MAGIC ##     COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University
# MAGIC ##     <url: https://github.com/CSSEGISandData/COVID-19>
# MAGIC 
# MAGIC git clone https://github.com/CSSEGISandData/COVID-19 /tmp/COVID-19

# COMMAND ----------

# DBTITLE 1,Add files to DBFS
# MAGIC %scala
# MAGIC 
# MAGIC dbutils.fs.cp("file:///tmp/COVID-19", "dbfs:/tmp/", recurse = true); // copy the COVID-19 data into dbfs

# COMMAND ----------

# DBTITLE 1,Daily Reports CSV Files
# MAGIC %fs
# MAGIC 
# MAGIC ls dbfs:/tmp/csse_covid_19_data/csse_covid_19_daily_reports/

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ingest Data into Delta

# COMMAND ----------

# DBTITLE 1,Ingest Data into Delta
# MAGIC %scala
# MAGIC 
# MAGIC import org.apache.spark.sql.functions.{input_file_name, current_timestamp, regexp_extract, to_date};
# MAGIC import org.apache.spark.sql.DataFrame;
# MAGIC import org.apache.spark.sql.types.{StructType, StructField, IntegerType, StringType, TimestampType, FloatType};
# MAGIC 
# MAGIC /*
# MAGIC 
# MAGIC   Load the John's Hopkin's COVID-19 data into a dataframe
# MAGIC 
# MAGIC */
# MAGIC 
# MAGIC var covid_source : String = "dbfs:/tmp/csse_covid_19_data/csse_covid_19_daily_reports/"; // covid source directory
# MAGIC var covid_files : Seq[String] = dbutils.fs.ls(covid_source).filter(_.path.contains(".csv")).map(_.path); // covid csv paths
# MAGIC var basename_regexp : String= "[^/]*(?=\\.[^.]+($|\\?))" // regex to extract the basename from a file (which contains the date)
# MAGIC 
# MAGIC /* specify the schema for the dataframe to ensure proper types */
# MAGIC var covid_schema = StructType(Array(
# MAGIC                       StructField("FIPS", StringType),
# MAGIC                       StructField("Admin2", StringType),
# MAGIC                       StructField("Province_State", StringType),
# MAGIC                       StructField("Country_Region", StringType),
# MAGIC                       StructField("Last_Update", TimestampType),
# MAGIC                       StructField("Lat", FloatType),
# MAGIC                       StructField("Long_", FloatType),
# MAGIC                       StructField("Confirmed", IntegerType),
# MAGIC                       StructField("Deaths", IntegerType),
# MAGIC                       StructField("Recovered", IntegerType),
# MAGIC                       StructField("Active", IntegerType),
# MAGIC                       StructField("Combined_Key", StringType),
# MAGIC                       StructField("Incidence_Rate", FloatType),
# MAGIC                       StructField("Case-Fatality_Ratio", FloatType)));
# MAGIC 
# MAGIC var df_covid : DataFrame = spark.read
# MAGIC                               .format("csv") // files are in csv format
# MAGIC                               .option("header", "true") // there's a header for each file
# MAGIC                               .schema(covid_schema) // specify schema to enforce types
# MAGIC                               .load(covid_files : _*) // load specified files using splat operator
# MAGIC                               .withColumn("file_source", input_file_name) // append the source file path
# MAGIC                               .withColumn("ingested_time", current_timestamp) // append the ingested time
# MAGIC                               .withColumn("date", to_date(regexp_extract($"file_source", basename_regexp, 0), "MM-dd-yyyy")) // append the date each file was created (date data is valid for)
# MAGIC                               .withColumnRenamed("Admin2", "County") // rename admin2 to county since it makes more sense
# MAGIC 
# MAGIC df_covid.write
# MAGIC         .format("delta")
# MAGIC         .mode("overwrite")
# MAGIC         .saveAsTable("jhu_covid_data")
# MAGIC 
# MAGIC display(df_covid); // display the dataframe

# COMMAND ----------

# DBTITLE 1,Retrieve Data from Delta Lake
# MAGIC %scala
# MAGIC 
# MAGIC import org.apache.spark.sql.DataFrame;
# MAGIC 
# MAGIC /*
# MAGIC   Setup a dataframe to read in data from
# MAGIC   a gold level table  
# MAGIC */
# MAGIC 
# MAGIC val df : DataFrame = spark.read
# MAGIC                           .format("delta")
# MAGIC                           .table("jhu_covid_data");
# MAGIC 
# MAGIC display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Get the Schema of the Data
# MAGIC 
# MAGIC Retrieve the schema of the data to get an idea as to what data types we are working with. This is used for featurizing the dataset.

# COMMAND ----------

# DBTITLE 1,Get the Schema of the Table
# MAGIC %sql
# MAGIC 
# MAGIC --
# MAGIC -- Get the schema of the columns
# MAGIC --
# MAGIC 
# MAGIC DESCRIBE jhu_covid_data;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Sanity Checks

# COMMAND ----------

# DBTITLE 1,Amount of Records Added Over Time
# MAGIC %sql
# MAGIC 
# MAGIC -- look at the amount of records added to the data set over time
# MAGIC 
# MAGIC SELECT date,
# MAGIC        count(date) as record_counts
# MAGIC FROM jhu_covid_data
# MAGIC GROUP BY date
# MAGIC ORDER BY date asc;

# COMMAND ----------

# DBTITLE 1,Number of Counties Added Over Time
# MAGIC %sql
# MAGIC 
# MAGIC -- look at the increase in administrators adding data into the system as a function of time during the outbreak
# MAGIC 
# MAGIC SELECT date,
# MAGIC        count(distinct county) as number_of_counties
# MAGIC FROM jhu_covid_data
# MAGIC GROUP BY date
# MAGIC ORDER BY date asc;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Dataframes for ML
# MAGIC 
# MAGIC Dataframes need to be featurized and split into train and test partitions for machine learning.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Clean the data

# COMMAND ----------

# DBTITLE 1,Remove Data That Doesn't Contain a FIPS
# MAGIC %scala
# MAGIC 
# MAGIC /*
# MAGIC   Remove censored data and cast data to proper data types
# MAGIC */
# MAGIC 
# MAGIC val cleaned_df : DataFrame = df.filter($"FIPS".isNotNull)
# MAGIC 
# MAGIC display(cleaned_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Featurize the Dataset

# COMMAND ----------

# DBTITLE 1,Create a ML Dataset
# MAGIC %python
# MAGIC 
# MAGIC from pyspark.ml.feature import StandardScaler, VectorAssembler, OneHotEncoder, StringIndexer
# MAGIC 
# MAGIC """
# MAGIC 
# MAGIC   Setup the datasets that will be used for this training example
# MAGIC 
# MAGIC """
# MAGIC 
# MAGIC df_covid = spark.read\
# MAGIC                 .format("delta")\
# MAGIC                 .table("jhu_covid_data")\
# MAGIC                 .na.drop() ## feed the temporary view into a python dataframe and drop all nulls
# MAGIC 
# MAGIC indexer = StringIndexer(inputCols=["Province_State", "Country_Region", "Combined_Key"],
# MAGIC                         outputCols=["category_province_state", "category_country_region", "category_combined_key"]) ## encode text into indices for k-means calculations
# MAGIC 
# MAGIC df_covid_indexed = indexer.fit(df_covid).transform(df_covid) ## generate the indexed dataframe
# MAGIC 
# MAGIC features = [x for x in df_covid_indexed.columns if x not in ["ingested_time", "file_source", "date", "FIPS", "Last_Update", "Province_State", "Country_Region", "Combined_Key", "County"]] ## use all features except metadata or those string indexed
# MAGIC 
# MAGIC assembler = VectorAssembler(inputCols = features,
# MAGIC                                 outputCol = "features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup the 70:30 train:test split

# COMMAND ----------

# DBTITLE 1,Create the train and test datasets
# MAGIC %python
# MAGIC 
# MAGIC from pyspark.sql import DataFrame;
# MAGIC 
# MAGIC """
# MAGIC   Separate the training and testing dataset into two dataframes
# MAGIC """
# MAGIC 
# MAGIC dfDataset = assembler.transform(df_covid_indexed)
# MAGIC trainingDF, testingDF = dfDataset.randomSplit([0.7, 0.3])

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training
# MAGIC 
# MAGIC For the purposes of this experiment, we will use MLFLOW to persist results and save models

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC import mlflow
# MAGIC 
# MAGIC """
# MAGIC Setup MLFlow Experiment ID to allow usage in Job Batches
# MAGIC """
# MAGIC 
# MAGIC current_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
# MAGIC 
# MAGIC mlflow.set_experiment(current_notebook_path+"_experiment")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## K-Means Model Training

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Function

# COMMAND ----------

# DBTITLE 1,K-Means Training Function
# MAGIC %python
# MAGIC 
# MAGIC from pyspark.ml.clustering import BisectingKMeans, BisectingKMeansModel
# MAGIC from pyspark.ml.evaluation import ClusteringEvaluator
# MAGIC from pyspark.sql import DataFrame
# MAGIC import mlflow
# MAGIC from typing import Tuple
# MAGIC 
# MAGIC """
# MAGIC   Setup K-Means modeling
# MAGIC """
# MAGIC 
# MAGIC def kMeansTrain(nCentroids : int,
# MAGIC                 seed : int,
# MAGIC                 dataset : DataFrame,
# MAGIC                 featuresCol : str = "features") -> Tuple[BisectingKMeansModel,float]:
# MAGIC   """
# MAGIC     Setup K-Means modeling
# MAGIC     
# MAGIC     @return Trained model
# MAGIC     @return Silhouete with squared euclidean distance 
# MAGIC     
# MAGIC     @param nCentroids   | number of centroids to cluster around
# MAGIC     @param seed          | random number seed
# MAGIC     @param dataset       | Spark DataFrame containing features
# MAGIC     @param featuresCol   | Name of the vectorized column
# MAGIC   """
# MAGIC   
# MAGIC   with mlflow.start_run() as run:
# MAGIC   
# MAGIC     mlflow.log_param("Number_Centroids", str(nCentroids))
# MAGIC     mlflow.log_metric("Training Data Rows", dataset.count())
# MAGIC     mlflow.log_param("seed", str(seed))
# MAGIC 
# MAGIC     ## Start up the bisecting k-means model
# MAGIC     bkm = BisectingKMeans()\
# MAGIC                       .setFeaturesCol(featuresCol)\
# MAGIC                       .setK(nCentroids)\
# MAGIC                       .setSeed(seed)\
# MAGIC                       .setPredictionCol("predictions")
# MAGIC 
# MAGIC     ## Start up the evaluator
# MAGIC     evaluator = ClusteringEvaluator()\
# MAGIC                       .setPredictionCol("predictions")
# MAGIC 
# MAGIC     ## Train a model
# MAGIC     model = bkm.fit(dataset)
# MAGIC 
# MAGIC     ## Make some predictions
# MAGIC     predictions = model.transform(dataset)
# MAGIC 
# MAGIC     ## Evaluate the clusters
# MAGIC     silhouette = evaluator.evaluate(predictions)
# MAGIC 
# MAGIC     ## Log some modeling metrics
# MAGIC     mlflow.log_metric("Silhouette", silhouette)
# MAGIC     mlflow.spark.log_model(model, f"K-Means_{nCentroids}")
# MAGIC 
# MAGIC   
# MAGIC     ## Return the class and silhouette
# MAGIC     return (model, silhouette)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train the K-Means Model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Hyperparameter Tuning for Number of Centroids

# COMMAND ----------

# DBTITLE 1,Tune the Number of Centroids
# MAGIC %python
# MAGIC 
# MAGIC """
# MAGIC   Tune a K-Means Model
# MAGIC """
# MAGIC 
# MAGIC ## Tune the K Means model by optimizing the number of centroids (hyperparameter tuning)
# MAGIC kMeansTuning = [(i, kMeansTrain(nCentroids = i, dataset = dfDataset, featuresCol = "features", seed = 1)) for i in range(2 ,15, 1)]
# MAGIC 
# MAGIC ## Return the results into a series of arrays
# MAGIC kMeansCosts = [(a[0], a[1][1]) for a in kMeansTuning]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Elbow Plot

# COMMAND ----------

# DBTITLE 1,Hierarchical Bisecting K-Means Cluster Tuning
# MAGIC %python
# MAGIC 
# MAGIC """
# MAGIC   Show the efffect of increasing the number of centroids
# MAGIC   for a K Means cluster
# MAGIC """
# MAGIC 
# MAGIC kMeansCostsDF = sc.parallelize(kMeansCosts)\
# MAGIC                       .toDF()\
# MAGIC                       .withColumnRenamed("_1", "Number of Centroids")\
# MAGIC                       .withColumnRenamed("_2", "Loss")
# MAGIC 
# MAGIC display(kMeansCostsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Label Dataset with Clusters

# COMMAND ----------

# DBTITLE 1,Augment Data with Cluster Label
# MAGIC %python
# MAGIC 
# MAGIC """
# MAGIC   Label the testing and training dataframes with the optimal clustering model
# MAGIC """
# MAGIC 
# MAGIC 
# MAGIC optimalClusterModel = kMeansTuning[9][1][0]
# MAGIC 
# MAGIC ## label the training and testing dataframes
# MAGIC clusteredtrainingDF = optimalClusterModel.transform(trainingDF)\
# MAGIC                             .withColumnRenamed("predictions", "cluster")
# MAGIC clusteredTestingDF = optimalClusterModel.transform(testingDF)\
# MAGIC                             .withColumnRenamed("predictions", "cluster")
# MAGIC 
# MAGIC clusteredtrainingDF.createOrReplaceTempView("clustered_training_df")
# MAGIC clusteredTestingDF.createOrReplaceTempView("clustered_testing_df")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Class Balance Between Training and Test Datasets

# COMMAND ----------

# DBTITLE 1,Validation DataFrame Class Balance
# MAGIC %sql
# MAGIC 
# MAGIC --
# MAGIC -- Check for class imbalance
# MAGIC --
# MAGIC 
# MAGIC SELECT "TRAINING" AS LABEL,
# MAGIC        CLUSTER,
# MAGIC        LOG(COUNT(*)) AS LOG_COUNT
# MAGIC FROM clustered_training_df
# MAGIC GROUP BY CLUSTER
# MAGIC UNION ALL
# MAGIC SELECT "TESTING" AS LABEL,
# MAGIC        CLUSTER,
# MAGIC        LOG(COUNT(*)) AS LOG_COUNT
# MAGIC FROM clustered_testing_df
# MAGIC GROUP BY CLUSTER
# MAGIC ORDER BY CLUSTER ASC;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Decision Tree Model Training

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Function

# COMMAND ----------

# DBTITLE 1,Decision Tree Training Function
import mlflow
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from typing import Tuple

def dtcTrain(p_max_depth : int,
             training_data : DataFrame,
             test_data : DataFrame,
             seed : int,
             featuresCol : str,
             labelCol : str) -> Tuple[int, float]:
  with mlflow.start_run() as run:
    # log some parameters
    mlflow.log_param("Maximum_depth", p_max_depth)
    mlflow.log_metric("Training Data Rows", training_data.count())
    mlflow.log_metric("Test Data Rows", test_data.count())
    
    # start the decision tree classifier
    dtc = DecisionTreeClassifier()\
                          .setFeaturesCol(featuresCol)\
                          .setLabelCol(labelCol)\
                          .setMaxDepth(p_max_depth)\
                          .setSeed(seed)\
                          .setPredictionCol("predictions")\
                          .setMaxBins(4000)
    
    # Start up the evaluator
    evaluator = MulticlassClassificationEvaluator()\
                      .setLabelCol("cluster")\
                      .setPredictionCol("predictions")

    # Train a model
    model = dtc.fit(training_data)

    # Make some predictions
    predictions = model.transform(test_data)

    # Evaluate the tree
    silhouette = evaluator.evaluate(predictions)
    
    # Log the accuracy
    mlflow.log_metric("F1", silhouette)
    
    # Log the feature importances
    mlflow.log_param("Feature Importances", model.featureImportances)
    
    # Log the model
    mlflow.spark.log_model(model, f"Decision_tree_{p_max_depth}")
    
    ## Return the class and silhouette
    return (model, silhouette)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train the Decision Tree

# COMMAND ----------

# MAGIC %md
# MAGIC #### Hyperparameter tuning for Max Depth

# COMMAND ----------

# DBTITLE 1,Tune Tree Depth
# MAGIC %python
# MAGIC 
# MAGIC """
# MAGIC Tune the max depth of the Decision tree
# MAGIC """
# MAGIC 
# MAGIC dtcTuning = [(i, dtcTrain(p_max_depth = i,
# MAGIC                           training_data = clusteredtrainingDF,
# MAGIC                           test_data = clusteredTestingDF,
# MAGIC                           seed = 1,
# MAGIC                           featuresCol = "features",
# MAGIC                           labelCol = "cluster"))
# MAGIC               for i in range(2, 15, 1)]
# MAGIC 
# MAGIC ## Return the results into a series of arrays
# MAGIC dtcF1 = [(a[0], a[1][1]) for a in dtcTuning]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Elbow Plot

# COMMAND ----------

# DBTITLE 1,Decision Tree Tuning
# MAGIC %python
# MAGIC 
# MAGIC """
# MAGIC   Show the efffect of increasing the max depth
# MAGIC   for a Decision Tree
# MAGIC """
# MAGIC 
# MAGIC dtcF1DF = sc.parallelize(dtcF1)\
# MAGIC                       .toDF()\
# MAGIC                       .withColumnRenamed("_1", "Max Depth")\
# MAGIC                       .withColumnRenamed("_2", "F1")
# MAGIC 
# MAGIC display(dtcF1DF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Multi-Dimensional Cross Validation

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC from pyspark.ml import Pipeline
# MAGIC from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# MAGIC 
# MAGIC """
# MAGIC Do a cross validation of the decision tree model
# MAGIC """
# MAGIC 
# MAGIC # Set the decision tree that will be optimized
# MAGIC dt = DecisionTreeClassifier()\
# MAGIC             .setFeaturesCol("features")\
# MAGIC             .setLabelCol("cluster")\
# MAGIC             .setMaxDepth(5)\
# MAGIC             .setSeed(1)\
# MAGIC             .setPredictionCol("predictions")\
# MAGIC             .setMaxBins(4000)
# MAGIC 
# MAGIC # Build the grid of different parameters
# MAGIC paramGrid = ParamGridBuilder() \
# MAGIC     .addGrid(dt.maxDepth, [5, 10, 15]) \
# MAGIC     .addGrid(dt.maxBins, [4000, 5000, 6000]) \
# MAGIC     .build()
# MAGIC 
# MAGIC # Generate an average F1 score for each prediction
# MAGIC evaluator = MulticlassClassificationEvaluator()\
# MAGIC                   .setLabelCol("cluster")\
# MAGIC                   .setPredictionCol("predictions")
# MAGIC 
# MAGIC # Build out the cross validation
# MAGIC crossval = CrossValidator(estimator = dt,
# MAGIC                           estimatorParamMaps = paramGrid,
# MAGIC                           evaluator = evaluator,
# MAGIC                           numFolds = 3)  
# MAGIC pipelineCV = Pipeline(stages=[crossval])
# MAGIC 
# MAGIC # Train the model using the pipeline, parameter grid, and preceding BinaryClassificationEvaluator
# MAGIC cvModel_u = pipelineCV.fit(clusteredtrainingDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Visualize the Optimal Decision Tree

# COMMAND ----------

# DBTITLE 1,Visualize Optimal Decision Tree
# MAGIC %python
# MAGIC 
# MAGIC """
# MAGIC Visualize the optimal decision tree
# MAGIC """
# MAGIC 
# MAGIC display(dtcTuning[5][1][0])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Forecasts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecast the tier/bracket a covid case will belong to

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC """
# MAGIC   Label the clustered predictions 
# MAGIC """
# MAGIC 
# MAGIC # choose the third decision tree model
# MAGIC optimalDTCModel = dtcTuning[5][1][0]
# MAGIC 
# MAGIC # create the segmentation DF
# MAGIC segmentationDF = optimalDTCModel.transform(testingDF)
# MAGIC 
# MAGIC display(segmentationDF);

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Deploy Model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup a pipeline with an assembler and decision tree

# COMMAND ----------

# DBTITLE 1,Create Deployment Model Pipeline
# MAGIC %python
# MAGIC 
# MAGIC from pyspark.ml import Pipeline, PipelineModel
# MAGIC 
# MAGIC """
# MAGIC   Combine the dataframe indexer and assembler with the decision tree
# MAGIC """
# MAGIC 
# MAGIC # Combine the existing models into a pipeline
# MAGIC deployment_ml_pipeline : Pipeline = Pipeline(stages = [indexer, assembler, optimalDTCModel])
# MAGIC 
# MAGIC # Create the pipeline transformer for the model by combining existing transformers
# MAGIC deployment_ml_pipeline_model : PipelineModel = deployment_ml_pipeline.fit(df_covid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use model to predict on an example dataframe

# COMMAND ----------

# DBTITLE 1,Forecast Using Deployment Model Pipeline
# MAGIC %python
# MAGIC 
# MAGIC from pyspark.sql import DataFrame
# MAGIC 
# MAGIC """
# MAGIC Forecast some results on the JHU Covid dataset
# MAGIC """
# MAGIC 
# MAGIC # classify the raw dataframe
# MAGIC df_deployed_pipeline_classification : DataFrame = deployment_ml_pipeline_model.transform(df_covid)
# MAGIC 
# MAGIC # visualize the results
# MAGIC display(df_deployed_pipeline_classification)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Register the model with MLFlow registry

# COMMAND ----------

# DBTITLE 1,Register Deployment Model with MLFlow
# MAGIC %python
# MAGIC 
# MAGIC import mlflow
# MAGIC from mlflow.models import ModelSignature
# MAGIC 
# MAGIC """
# MAGIC Log the mlflow model to the registry
# MAGIC """
# MAGIC 
# MAGIC model_version_major = 1
# MAGIC model_version_minor = 2
# MAGIC 
# MAGIC with mlflow.start_run() as run:
# MAGIC   # get the dataframe signature for the model
# MAGIC   _signature = mlflow.models.infer_signature(df_covid\
# MAGIC                                               .drop("ingested_time")\
# MAGIC                                               .drop("date")\
# MAGIC                                               .drop("Last_Update"),
# MAGIC                                             df_deployed_pipeline_classification\
# MAGIC                                               .drop("features")\
# MAGIC                                               .drop("ingested_time")\
# MAGIC                                               .drop("date")\
# MAGIC                                               .drop("Last_Update")\
# MAGIC                                               .drop("probability")\
# MAGIC                                               .drop("rawPrediction"))
# MAGIC   # Log the model
# MAGIC   mlflow.spark.log_model(spark_model = deployment_ml_pipeline_model,
# MAGIC                          signature = _signature,
# MAGIC                          registered_model_name = "jhu_covid",
# MAGIC                          artifact_path = f"pipeline_model_v{model_version_major}.{model_version_minor}"
# MAGIC                         )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Access the model via MLFlow Python API

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Prepare data

# COMMAND ----------

# DBTITLE 1,Create JSON Data
# MAGIC %python
# MAGIC 
# MAGIC """
# MAGIC Generate JSON records
# MAGIC """
# MAGIC 
# MAGIC json_records = df_covid.limit(100).toPandas().to_json(orient='split')
# MAGIC print(json_records)

# COMMAND ----------

# DBTITLE 1,Read Data as Pandas Dataframe
import pandas as pd

"""
Read JSON Data as a Pandas dataframe
"""

pd.read_json(json_records, orient = 'split')

# COMMAND ----------

# DBTITLE 1,Access Deployed Model using Python API
# MAGIC %python
# MAGIC 
# MAGIC import os
# MAGIC import requests
# MAGIC import pandas as pd
# MAGIC 
# MAGIC """
# MAGIC Example scoring model from MLFlow UI
# MAGIC Need to specify bearer token to connect to current MLFlow deployment
# MAGIC """
# MAGIC 
# MAGIC # define the model scoring function
# MAGIC def score_model(dataset: pd.DataFrame):
# MAGIC   url = 'https://field-eng.cloud.databricks.com/model/jhu_covid/3/invocations'
# MAGIC   headers = {'Authorization': f'Bearer {dbutils.secrets.get("ml-ml", "TOKEN")}'}
# MAGIC   data_json = json_records
# MAGIC   response = requests.request(method='POST', headers=headers, url=url, json=data_json)
# MAGIC   if response.status_code != 200:
# MAGIC     raise Exception(f'Request failed with status {response.status_code}, {response.text}')
# MAGIC   return response.json()
# MAGIC 
# MAGIC # score the model
# MAGIC score_model(df_dataset)