from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def main():

    # ==============================
    # 1️⃣ Spark Session
    # ==============================
    spark = SparkSession.builder \
        .appName("Healthcare_Disease_Prediction_Optimized") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("Stage 1: Spark Session Started")

    # ==============================
    # 2️⃣ Load Dataset
    # ==============================
    input_path = "s3a://health-care-2026/healthcare_disease_prediction_50_records.csv"

    df = spark.read.option("header", "true") \
                   .option("inferSchema", "true") \
                   .csv(input_path)

    print("Stage 2: Dataset Loaded")

    # ==============================
    # 3️⃣ Data Cleaning
    # ==============================
    df = df.dropna()
    print("Stage 3: Data Cleaning Completed")

    # ==============================
    # 4️⃣ Handle Imbalanced Data (Class Weights)
    # ==============================
    print("Stage 4: Handling Imbalanced Data")

    class_counts = df.groupBy("disease").count().collect()
    total = sum(row["count"] for row in class_counts)

    weights = {}
    for row in class_counts:
        weights[row["disease"]] = total / (2 * row["count"])

    df = df.withColumn(
        "classWeightCol",
        when(col("disease") == 0, weights[0]).otherwise(weights[1])
    )

    print("Class Weights Applied:", weights)

    # ==============================
    # 5️⃣ Feature Engineering
    # ==============================
    feature_columns = ["age", "bmi", "blood_pressure",
                       "glucose_level", "cholesterol"]

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
    print("Stage 4: Feature Engineering Completed")
    # ==============================
    # 6️⃣ Train-Test Split
    # ==============================
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print("Stage 5: Train-Test Split Completed")
    print("Training Count :", train_df.count())
    print("Testing Count  :", test_df.count())

    # ==============================
    # 7️⃣ Random Forest Model (With Weight)
    # ==============================
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="disease",
        weightCol="classWeightCol"
    )

    pipeline = Pipeline(stages=[assembler, rf])

    # ==============================
    # 8️⃣ Hyperparameter Tuning
    # ==============================
    print("Stage 6: Hyperparameter Tuning Started")

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="disease",
        predictionCol="prediction",
        metricName="recallByLabel",
        metricLabel=1.0  # Focus on disease recall
    )

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3
    )

    model = crossval.fit(train_df)

    print("Stage 6: Hyperparameter Tuning Completed")

    # ==============================
    # 9️⃣ Predictions
    # ==============================
    predictions = model.transform(test_df)

    print("Stage 7: Predictions Completed")
    predictions.select("disease", "prediction").show()

    # ==============================
    # 🔟 Confusion Matrix
    # ==============================
    print("Stage 8: Confusion Matrix")
    predictions.groupBy("disease", "prediction").count().show()

    # ==============================
    # 1️⃣1️⃣ Final Evaluation Metrics
    # ==============================

    accuracy = MulticlassClassificationEvaluator(
        labelCol="disease",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol="disease",
        predictionCol="prediction",
        metricName="f1"
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="disease",
        predictionCol="prediction",
        metricName="recallByLabel",
        metricLabel=1.0
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol="disease",
        predictionCol="prediction",
        metricName="precisionByLabel",
        metricLabel=1.0
    ).evaluate(predictions)

    print("Stage 9: Model Evaluation Completed")
    print("Accuracy          :", accuracy)
    print("F1 Score          :", f1)
    print("Disease Recall    :", recall)
    print("Disease Precision :", precision)

    # ==============================
    # 1️⃣2️⃣ Save Predictions
    # ==============================
    output_path = "s3a://health-care-2026/optimized_healthcare_predictions"

    predictions.write.mode("overwrite").parquet(output_path)

    print("Stage 10: Optimized Predictions Saved to S3")

    spark.stop()
    print("Spark Session Stopped Successfully")


if __name__ == "__main__":
    main()
