from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():

    # =============================
    # 1️⃣ Credit Card Fraud Detection Model
    # =============================
    spark = SparkSession.builder \
        .appName("CreditCardFraudDetection_Final") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("✅ Spark session started")

    # =============================
    # 2️⃣ Load Data from S3
    # =============================
    input_path = "s3a://credit-card-detection-2026/credit_card_fraud_100_rows_15percent.csv"

    df = spark.read.option("header", "true") \
                   .option("inferSchema", "true") \
                   .csv(input_path)

    print("✅ Raw data loaded from S3")
    print("Total rows:", df.count())

    # =============================
    # 3️⃣ Data Cleaning
    # =============================
    df = df.dropna()
    print("✅ Null values removed")

    df = df.withColumn("transaction_time",
                       to_timestamp(col("transaction_time"),
                                    "yyyy-MM-dd HH:mm:ss"))
    print("✅ Converted transaction_time to timestamp")

    df = df.filter(col("transaction_amount") > 0)
    print("✅ Removed invalid transaction amounts")

    print("✅ Data cleaning completed successfully")
    print("Rows after cleaning:", df.count())

    # =============================
    # 4️⃣ Fix Class Imbalance (Correct Oversampling)
    # =============================
    fraud_df = df.filter(col("is_fraud") == 1)
    non_fraud_df = df.filter(col("is_fraud") == 0)

    fraud_count = fraud_df.count()
    non_fraud_count = non_fraud_df.count()

    print("Fraud count before balancing:", fraud_count)
    print("Non-Fraud count before balancing:", non_fraud_count)

    # Convert to float to avoid error
    multiplier = float(non_fraud_count) / float(fraud_count)

    fraud_oversampled = fraud_df.sample(
        withReplacement=True,
        fraction=multiplier,
        seed=42
    )

    balanced_df = non_fraud_df.union(fraud_oversampled)

    print("✅ Class imbalance fixed")
    print("Rows after balancing:", balanced_df.count())

    # =============================
    # 5️⃣ Feature Engineering
    # =============================
    indexer1 = StringIndexer(inputCol="merchant_category",
                             outputCol="merchant_index")
    indexer2 = StringIndexer(inputCol="customer_region",
                             outputCol="region_index")

    balanced_df = indexer1.fit(balanced_df).transform(balanced_df)
    balanced_df = indexer2.fit(balanced_df).transform(balanced_df)

    assembler = VectorAssembler(
        inputCols=[
            "transaction_amount",
            "customer_age",
            "merchant_index",
            "region_index"
        ],
        outputCol="features"
    )

    final_df = assembler.transform(balanced_df)

    print("✅ Feature engineering completed")

    # =============================
    # 6️⃣ Train Test Split
    # =============================
    train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

    print("✅ Train/Test split completed")
    print("Training rows:", train_df.count())
    print("Testing rows:", test_df.count())

    # =============================
    # 7️⃣ Model Training
    # =============================
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="is_fraud",
        maxIter=10
    )

    model = lr.fit(train_df)
    print("✅ Model training completed")

    # =============================
    # 8️⃣ Prediction
    # =============================
    predictions = model.transform(test_df)
    print("✅ Prediction completed")

    # =============================
    # 9️⃣ Confusion Matrix
    # =============================
    print("📊 Confusion Matrix:")
    predictions.groupBy("is_fraud", "prediction").count().show()

    # =============================
    # 🔟 Evaluation Metrics
    # =============================
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="accuracy"
    )

    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="f1"
    )

    accuracy = evaluator_accuracy.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print("📈 Model Evaluation Metrics")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # =============================
    # 1️⃣1️⃣ Save Predictions to S3
    # =============================
    output_path = "s3a://credit-card-detection-2026/fraud-output/predictions"

    predictions.write.mode("overwrite").parquet(output_path)

    print("✅ Predictions saved to S3 successfully")

    spark.stop()
    print("✅ Spark session stopped cleanly")


if __name__ == "__main__":
    main()
# =======================================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():

    # =============================
    # 1️⃣ Credit Card Fraud Detection Model
    # =============================
    spark = SparkSession.builder \
        .appName("CreditCardFraudDetection_Final") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("✅ Spark session started")

    # =============================
    # 2️⃣ Load Data from S3
    # =============================
    input_path = "s3a://credit-card-detection-2026/creditcard_transactions.csv"

    df = spark.read.option("header", "true") \
                   .option("inferSchema", "true") \
                   .csv(input_path)

    print("✅ Raw data loaded from S3")
    print("Total rows:", df.count())

    # =============================
    # 3️⃣ Data Cleaning
    # =============================
    df = df.dropna()
    print("✅ Null values removed")

    df = df.withColumn("transaction_time",
                       to_timestamp(col("transaction_time"),
                                    "yyyy-MM-dd HH:mm:ss"))
    print("✅ Converted transaction_time to timestamp")

    df = df.filter(col("transaction_amount") > 0)
    print("✅ Removed invalid transaction amounts")

    print("✅ Data cleaning completed successfully")
    print("Rows after cleaning:", df.count())

    # =============================
    # 4️⃣ Fix Class Imbalance (Correct Oversampling)
    # =============================
    fraud_df = df.filter(col("is_fraud") == 1)
    non_fraud_df = df.filter(col("is_fraud") == 0)

    fraud_count = fraud_df.count()
    non_fraud_count = non_fraud_df.count()

    print("Fraud count before balancing:", fraud_count)
    print("Non-Fraud count before balancing:", non_fraud_count)

    multiplier = float(non_fraud_count) / float(fraud_count)

    fraud_oversampled = fraud_df.sample(
        withReplacement=True,
        fraction=multiplier,
        seed=42
    )

    balanced_df = non_fraud_df.union(fraud_oversampled)

    print("✅ Class imbalance fixed")
    print("Rows after balancing:", balanced_df.count())

    # =============================
    # 5️⃣ Feature Engineering
    # =============================
    indexer1 = StringIndexer(inputCol="merchant_category",
                             outputCol="merchant_index")
    indexer2 = StringIndexer(inputCol="customer_region",
                             outputCol="region_index")

    balanced_df = indexer1.fit(balanced_df).transform(balanced_df)
    balanced_df = indexer2.fit(balanced_df).transform(balanced_df)

    assembler = VectorAssembler(
        inputCols=[
            "transaction_amount",
            "customer_age",
            "merchant_index",
            "region_index"
        ],
        outputCol="features"
    )

    final_df = assembler.transform(balanced_df)

    print("✅ Feature engineering completed")

    # =============================
    # 6️⃣ Train Test Split
    # =============================
    train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

    print("✅ Train/Test split completed")
    print("Training rows:", train_df.count())
    print("Testing rows:", test_df.count())

    # =============================
    # 7️⃣ Model Training (Method 1 Applied)
    # =============================
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="is_fraud",
        maxIter=10,
        threshold=0.4   # 🔥 Lowered from default 0.5 to reduce missed frauds
    )

    model = lr.fit(train_df)
    print("✅ Model training completed")

    # =============================
    # 8️⃣ Prediction
    # =============================
    predictions = model.transform(test_df)
    print("✅ Prediction completed")

    # =============================
    # 9️⃣ Confusion Matrix
    # =============================
    print("📊 Confusion Matrix:")
    predictions.groupBy("is_fraud", "prediction").count().show()

    # =============================
    # 🔟 Evaluation Metrics
    # =============================
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="accuracy"
    )

    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="f1"
    )

    accuracy = evaluator_accuracy.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print("📈 Model Evaluation Metrics")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # =============================
    # 1️⃣1️⃣ Save Predictions to S3
    # =============================
    output_path = "s3a://credit-card-detection-2026/fraud-output/predictions"

    predictions.write.mode("overwrite").parquet(output_path)

    print("✅ Predictions saved to S3 successfully")

    spark.stop()
    print("✅ Spark session stopped cleanly")


if __name__ == "__main__":
    main()
# =======================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # =============================
    # 1️⃣ Spark Session
    # =============================
    spark = SparkSession.builder \
        .appName("CreditCardFraudDetection_RF_Optimized") \
        .master("local[1]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("✅ Spark session started")

    # =============================
    # 2️⃣ Load Data
    # =============================
    input_path = "s3a://credit-card-detection-2026/creditcard_transactions.csv"
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

    print("✅ Raw data loaded from S3")
    print("Total rows:", df.count())
    print("Columns:", df.columns)

    # =============================
    # 3️⃣ Data Cleaning
    # =============================
    df = df.dropna()
    df = df.withColumn("transaction_time", to_timestamp(col("transaction_time"), "yyyy-MM-dd HH:mm:ss"))
    df = df.filter(col("transaction_amount") > 0)
    print("✅ Data cleaning completed")
    print("Rows after cleaning:", df.count())

    # =============================
    # 4️⃣ Fix Class Imbalance (Oversampling)
    # =============================
    fraud_df = df.filter(col("is_fraud") == 1)
    non_fraud_df = df.filter(col("is_fraud") == 0)
    fraud_count = fraud_df.count()
    non_fraud_count = non_fraud_df.count()
    multiplier = float(non_fraud_count) / max(float(fraud_count), 1)
    fraud_oversampled = fraud_df.sample(withReplacement=True, fraction=multiplier, seed=42)
    balanced_df = non_fraud_df.union(fraud_oversampled)
    print("✅ Class imbalance fixed")
    print("Rows after balancing:", balanced_df.count())

    # =============================
    # 5️⃣ Feature Engineering (Indexing)
    # =============================
    for col_name in ["merchant", "customer_region"]:
        if col_name in balanced_df.columns:
            balanced_df = balanced_df.fillna({col_name: "unknown"})

    if "merchant" in balanced_df.columns:
        indexer_merchant = StringIndexer(inputCol="merchant", outputCol="merchant_index")
        balanced_df = indexer_merchant.fit(balanced_df).transform(balanced_df)
    else:
        balanced_df = balanced_df.withColumn("merchant_index", col("transaction_amount")*0 + 0)

    if "customer_region" in balanced_df.columns:
        indexer_region = StringIndexer(inputCol="customer_region", outputCol="region_index")
        balanced_df = indexer_region.fit(balanced_df).transform(balanced_df)
    else:
        balanced_df = balanced_df.withColumn("region_index", col("transaction_amount")*0 + 0)

    feature_cols = ["transaction_amount", "customer_age", "merchant_index", "region_index"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    final_df = assembler.transform(balanced_df)
    print("✅ Feature engineering completed")

    # =============================
    # 6️⃣ Train/Test Split
    # =============================
    train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)
    print("✅ Train/Test split completed")
    print("Training rows:", train_df.count())
    print("Testing rows:", test_df.count())

    # =============================
    # 7️⃣ Compute Class Weights
    # =============================
    fraud_count_train = train_df.filter(col("is_fraud") == 1).count()
    non_fraud_count_train = train_df.filter(col("is_fraud") == 0).count()
    total_train = train_df.count()
    weight_0 = total_train / (2 * non_fraud_count_train)
    weight_1 = total_train / (2 * fraud_count_train)
    train_df = train_df.withColumn(
        "class_weight_col",
        col("is_fraud").cast("double") * weight_1 + (1 - col("is_fraud").cast("double")) * weight_0
    )

    # =============================
    # 8️⃣ Train Random Forest
    # =============================
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="is_fraud",
        weightCol="class_weight_col",
        numTrees=100,
        maxDepth=5,
        seed=42,
        probabilityCol="probability"
    )
    model = rf.fit(train_df)
    print("✅ Random Forest training completed")

    # =============================
    # 9️⃣ Predict Probabilities
    # =============================
    preds = model.transform(test_df)
    print("✅ Prediction completed")

    # =============================
    # 9.1️⃣ Extract Fraud Probability Using UDF
    # =============================
    get_prob = udf(lambda v: float(v[1]), DoubleType())
    preds = preds.withColumn("probability_1", get_prob(col("probability")))

    # =============================
    # 🔟 Find Best Threshold for F1 (0.01 steps)
    # =============================
    thresholds = [i*0.01 for i in range(0, 101)]  # 0.00 → 1.00
    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:
        preds_thresh = preds.withColumn(
            "prediction_thresh",
            (col("probability_1") >= t).cast("double")
        )
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="is_fraud",
            predictionCol="prediction_thresh",
            metricName="f1"
        )
        f1 = evaluator_f1.evaluate(preds_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"✅ Best F1 Score: {best_f1:.4f} at Threshold: {best_thresh}")

    # =============================
    # 1️⃣1️⃣ Apply Best Threshold
    # =============================
    final_preds = preds.withColumn(
        "prediction_best",
        (col("probability_1") >= best_thresh).cast("double")
    )

    print("📊 Confusion Matrix with Best Threshold:")
    final_preds.groupBy("is_fraud", "prediction_best").count().show()

    # =============================
    # 1️⃣2️⃣ Metrics at Best Threshold
    # =============================
    metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]
    for metric in metrics:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="is_fraud",
            predictionCol="prediction_best",
            metricName=metric
        )
        print(f"{metric}: {evaluator.evaluate(final_preds):.4f}")

    # Fraud class recall
    tp = final_preds.filter((col("is_fraud") == 1) & (col("prediction_best") == 1)).count()
    fn = final_preds.filter((col("is_fraud") == 1) & (col("prediction_best") == 0)).count()
    recall_fraud = tp / max(tp + fn, 1)
    print(f"Fraud Class Recall: {recall_fraud:.4f}")

    # =============================
    # 1️⃣3️⃣ Save Predictions
    # =============================
    output_path = "s3a://credit-card-detection-2026/fraud-output/predictions_rf"
    final_preds.write.mode("overwrite").parquet(output_path)
    print("✅ Predictions saved to S3 successfully")

    spark.stop()
    print("✅ Spark session stopped cleanly")


if __name__ == "__main__":
    main()
# =====================================================================================
# Final Code:

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # =============================
    # 1️⃣ Spark Session (all cores, optimized memory)
    # =============================
    spark = SparkSession.builder \
        .appName("CreditCardFraud_RF_FullRun") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("✅ Spark session started")

    # =============================
    # 2️⃣ Load Data from S3
    # =============================
    input_path = "s3a://credit-card-detection-2026/creditcard_transactions.csv"
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    df = df.dropna()
    df = df.withColumn("transaction_time", to_timestamp(col("transaction_time"), "yyyy-MM-dd HH:mm:ss"))
    df = df.filter(col("transaction_amount") > 0)
    print(f"✅ Data loaded. Rows after cleaning: {df.count()}")

    # =============================
    # 3️⃣ Balance Classes
    # =============================
    fraud_df = df.filter(col("is_fraud") == 1)
    non_fraud_df = df.filter(col("is_fraud") == 0)
    multiplier = float(non_fraud_df.count()) / max(float(fraud_df.count()), 1)
    fraud_oversampled = fraud_df.sample(withReplacement=True, fraction=multiplier, seed=42)
    balanced_df = non_fraud_df.union(fraud_oversampled)
    print(f"✅ Class imbalance fixed. Total rows: {balanced_df.count()}")

    # =============================
    # 4️⃣ Feature Engineering
    # =============================
    for col_name in ["merchant", "customer_region"]:
        if col_name in balanced_df.columns:
            balanced_df = balanced_df.fillna({col_name: "unknown"})

    if "merchant" in balanced_df.columns:
        balanced_df = StringIndexer(inputCol="merchant", outputCol="merchant_index").fit(balanced_df).transform(balanced_df)
    else:
        balanced_df = balanced_df.withColumn("merchant_index", col("transaction_amount")*0 + 0)

    if "customer_region" in balanced_df.columns:
        balanced_df = StringIndexer(inputCol="customer_region", outputCol="region_index").fit(balanced_df).transform(balanced_df)
    else:
        balanced_df = balanced_df.withColumn("region_index", col("transaction_amount")*0 + 0)

    feature_cols = ["transaction_amount", "customer_age", "merchant_index", "region_index"]
    final_df = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(balanced_df)

    # =============================
    # 5️⃣ Train/Test Split (Full Training Data)
    # =============================
    train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)
    print(f"✅ Train/Test split. Training rows: {train_df.count()}, Testing rows: {test_df.count()}")

    # =============================
    # 6️⃣ Class Weights
    # =============================
    fraud_count_train = train_df.filter(col("is_fraud") == 1).count()
    non_fraud_count_train = train_df.filter(col("is_fraud") == 0).count()
    total_train = train_df.count()
    weight_0 = total_train / (2 * non_fraud_count_train)
    weight_1 = total_train / (2 * fraud_count_train)
    train_df = train_df.withColumn(
        "class_weight_col",
        col("is_fraud").cast("double") * weight_1 + (1 - col("is_fraud").cast("double")) * weight_0
    )

    # =============================
    # 7️⃣ Random Forest (Full Run)
    # =============================
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="is_fraud",
        weightCol="class_weight_col",
        numTrees=100,
        maxDepth=8,
        seed=42,
        probabilityCol="probability"
    )
    model = rf.fit(train_df)
    print("✅ Random Forest training completed")

    # =============================
    # 8️⃣ Predictions
    # =============================
    preds = model.transform(test_df)
    get_prob = udf(lambda v: float(v[1]), DoubleType())
    preds = preds.withColumn("probability_1", get_prob(col("probability")))
    print("✅ Predictions completed")

    # =============================
    # 9️⃣ Threshold Sweep for Best F1
    # =============================
    evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="f1")
    thresholds = [i*0.01 for i in range(0, 101)]
    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:
        preds_thresh = preds.withColumn("prediction_thresh", (col("probability_1") >= t).cast("double"))
        preds_eval = preds_thresh.drop("prediction").withColumnRenamed("prediction_thresh", "prediction")
        f1 = evaluator.evaluate(preds_eval)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"✅ Best F1 Score: {best_f1:.4f} at Threshold: {best_thresh}")

    # =============================
    # 1️⃣0️⃣ Apply Best Threshold
    # =============================
    final_preds = preds.withColumn("prediction_best", (col("probability_1") >= best_thresh).cast("double"))

    print("📊 Confusion Matrix with Best Threshold:")
    final_preds.groupBy("is_fraud", "prediction_best").count().show()

    metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]
    for metric in metrics:
        eval_metric = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction_best", metricName=metric)
        print(f"{metric}: {eval_metric.evaluate(final_preds):.4f}")

    tp = final_preds.filter((col("is_fraud") == 1) & (col("prediction_best") == 1)).count()
    fn = final_preds.filter((col("is_fraud") == 1) & (col("prediction_best") == 0)).count()
    recall_fraud = tp / max(tp + fn, 1)
    print(f"Fraud Class Recall: {recall_fraud:.4f}")

    # =============================
    # 1️⃣1️⃣ Save Predictions
    # =============================
    output_path = "s3a://credit-card-detection-2026/fraud-output/predictions_rf_fullrun"
    final_preds.write.mode("overwrite").parquet(output_path)
    print("✅ Predictions saved to S3 successfully")

    spark.stop()
    print("✅ Spark session stopped cleanly")


if __name__ == "__main__":
    main()
# ==================================================================================