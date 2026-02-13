from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

print("===== REAL-TIME CUSTOMER CHURN PREDICTION (ULTRA-LIGHT + S3) =====")

# 1️⃣ Spark Session
spark = SparkSession.builder \
    .appName("CustomerChurnPrediction_RealTime") \
    .master("local[1]") \
    .config("spark.driver.memory", "500m") \
    .config("spark.sql.shuffle.partitions", "1") \
    .config("spark.driver.host", "0.0.0.0") \
    .config("spark.network.timeout", "60s") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.us-east-1.amazonaws.com") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✔ Spark session started")

# 2️⃣ Read CSV from S3
input_path = "s3a://spark-churn-2026/raw/churn.csv"
df = spark.read.option("header","true").option("inferSchema","true").csv(input_path)
df = df.cache()
print("✔ Raw data loaded and cached")
df.show(5)

# 3️⃣ Data Cleaning
df = df.dropna()

numeric_cols = ["senior_citizen","tenure_months","monthly_charges","total_charges","avg_monthly_usage","complaints"]
for c in numeric_cols:
    df = df.withColumn(c, col(c).cast("double"))

df = df.withColumn("churn_label", when(col("churn")=="Y", 1).otherwise(0))
print("✔ Data cleaning completed")

# 4️⃣ Feature Engineering
categorical_cols = ["gender","partner","dependents","phone_service","internet_service","contract_type","payment_method"]

for c in categorical_cols:
    indexer = StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
    df = indexer.fit(df).transform(df)

assembler = VectorAssembler(
    inputCols=[c+"_idx" for c in categorical_cols] + numeric_cols,
    outputCol="features"
)
df = assembler.transform(df)
print("✔ Feature engineering completed")

# 5️⃣ Train Logistic Regression
lr = LogisticRegression(labelCol="churn_label", featuresCol="features", maxIter=10)
model = lr.fit(df)
print("✔ Model trained")

# 6️⃣ Predictions
predictions = model.transform(df)
predictions.select("customer_id","prediction","probability","churn_label").show(truncate=False)

# 7️⃣ Evaluation Metrics
# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="churn_label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator_acc.evaluate(predictions)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Confusion Matrix
y_true_pred = predictions.select("churn_label","prediction").rdd.map(lambda row: (row[0], row[1]))
conf_matrix = y_true_pred \
    .map(lambda x: ((x[0], x[1]), 1)) \
    .reduceByKey(lambda a,b: a+b) \
    .collect()
print("\nConfusion Matrix:")
print(conf_matrix)  # [(true,pred), count] format

# Classification Report (precision, recall, F1)
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="churn_label", predictionCol="prediction", metricName="weightedPrecision"
)
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="churn_label", predictionCol="prediction", metricName="weightedRecall"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="churn_label", predictionCol="prediction", metricName="f1"
)
print(f"\nClassification Report:")
print(f"Precision: {evaluator_precision.evaluate(predictions):.4f}")
print(f"Recall:    {evaluator_recall.evaluate(predictions):.4f}")
print(f"F1 Score:  {evaluator_f1.evaluate(predictions):.4f}")

# 8️⃣ Write Predictions to S3
output_path = "s3a://spark-churn-2026/output/churn_predictions_ultralight"
predictions.write.mode("overwrite").parquet(output_path)
print("✔ Predictions written to S3")

# 9️⃣ Stop Spark
spark.stop()
print("===== PROJECT COMPLETED SUCCESSFULLY =====")

# ==============================================================================

# Test data set code

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when
# from pyspark.ml.feature import StringIndexer, VectorAssembler
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# print("===== REAL-TIME CUSTOMER CHURN PREDICTION (ULTRA-LIGHT + S3) =====")

# # 1️⃣ Spark Session
# spark = SparkSession.builder \
#     .appName("CustomerChurnPrediction_RealTime") \
#     .master("local[1]") \
#     .config("spark.driver.memory", "500m") \
#     .config("spark.sql.shuffle.partitions", "1") \
#     .config("spark.driver.host", "0.0.0.0") \
#     .config("spark.network.timeout", "60s") \
#     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
#     .config("spark.hadoop.fs.s3a.aws.credentials.provider",
#             "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
#     .config("spark.hadoop.fs.s3a.endpoint", "s3.us-east-1.amazonaws.com") \
#     .getOrCreate()

# spark.sparkContext.setLogLevel("ERROR")
# print("✔ Spark session started")

# # 2️⃣ Read CSV from S3
# input_path = "s3a://spark-churn-2026/raw/churn.csv"
# df = spark.read.option("header","true").option("inferSchema","true").csv(input_path)
# df = df.cache()
# print("✔ Raw data loaded and cached")
# df.show(5)

# # 3️⃣ Data Cleaning
# df = df.dropna()
# numeric_cols = ["senior_citizen","tenure_months","monthly_charges","total_charges","avg_monthly_usage","complaints"]
# for c in numeric_cols:
#     df = df.withColumn(c, col(c).cast("double"))
# df = df.withColumn("churn_label", when(col("churn")=="Y", 1).otherwise(0))
# print("✔ Data cleaning completed")

# # 4️⃣ Feature Engineering
# categorical_cols = ["gender","partner","dependents","phone_service","internet_service","contract_type","payment_method"]
# for c in categorical_cols:
#     indexer = StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
#     df = indexer.fit(df).transform(df)

# assembler = VectorAssembler(
#     inputCols=[c+"_idx" for c in categorical_cols] + numeric_cols,
#     outputCol="features"
# )
# df = assembler.transform(df)
# print("✔ Feature engineering completed")

# # 5️⃣ Train/Test Split
# train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
# print(f"✔ Data split into train ({train_df.count()} rows) and test ({test_df.count()} rows)")

# # 6️⃣ Train Logistic Regression on train set
# lr = LogisticRegression(labelCol="churn_label", featuresCol="features", maxIter=10)
# model = lr.fit(train_df)
# print("✔ Model trained on training set")

# # 7️⃣ Predictions on test set for evaluation
# pred_test = model.transform(test_df)
# pred_test.select("customer_id","prediction","probability","churn_label").show(5, truncate=False)

# # 8️⃣ Evaluation Metrics
# evaluator_acc = MulticlassClassificationEvaluator(labelCol="churn_label", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator_acc.evaluate(pred_test)
# print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")

# # Confusion Matrix
# y_true_pred = pred_test.select("churn_label","prediction").rdd.map(lambda row: (row[0], row[1]))
# conf_matrix = y_true_pred.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda a,b: a+b).collect()
# print("\nConfusion Matrix (true_label, predicted_label): count")
# print(conf_matrix)

# # Classification Report
# evaluator_precision = MulticlassClassificationEvaluator(labelCol="churn_label", predictionCol="prediction", metricName="weightedPrecision")
# evaluator_recall = MulticlassClassificationEvaluator(labelCol="churn_label", predictionCol="prediction", metricName="weightedRecall")
# evaluator_f1 = MulticlassClassificationEvaluator(labelCol="churn_label", predictionCol="prediction", metricName="f1")
# print(f"\nClassification Report:")
# print(f"Precision: {evaluator_precision.evaluate(pred_test):.4f}")
# print(f"Recall:    {evaluator_recall.evaluate(pred_test):.4f}")
# print(f"F1 Score:  {evaluator_f1.evaluate(pred_test):.4f}")

# # 9️⃣ Predictions on full data for S3 storage
# predictions_full = model.transform(df)
# predictions_full.select("customer_id","prediction","probability","churn_label").show(5, truncate=False)

# output_path = "s3a://spark-churn-2026/output/churn_predictions_ultralight"
# predictions_full.write.mode("overwrite").parquet(output_path)
# print("✔ Predictions written to S3")

# # 1️⃣0️⃣ Stop Spark
# spark.stop()
# print("===== PROJECT COMPLETED SUCCESSFULLY =====")
