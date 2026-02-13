from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    StringIndexer,
    IndexToString
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def main():

    # ==================================================
    # 1️⃣ Start Spark Session
    # ==================================================
    print("\n🚀 Starting Spark Session...\n")

    spark = SparkSession.builder \
        .appName("Sentiment Analysis - Final Production Version") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .getOrCreate()

    print("✅ Spark session started successfully.\n")

    # ==================================================
    # 2️⃣ Load Dataset
    # ==================================================
    print("📥 Loading dataset from S3...")

    df = spark.read.csv(
        "s3a://social-media-2026/social_media_sentiment_50_records.csv",
        header=True,
        inferSchema=True
    ).dropna()

    total_records = df.count()
    print(f"✅ Dataset loaded successfully. Total records: {total_records}\n")

    # ==================================================
    # 3️⃣ Label Encoding (Safe)
    # ==================================================
    print("🔄 Encoding sentiment labels...")

    label_indexer = StringIndexer(
        inputCol="sentiment",
        outputCol="label",
        handleInvalid="keep"
    )

    label_model = label_indexer.fit(df)
    df = label_model.transform(df)

    print("✅ Sentiment labels encoded successfully.\n")

    # ==================================================
    # 4️⃣ Train-Test Split
    # ==================================================
    print("🔀 Splitting dataset into training and testing sets...")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print(f"✅ Data split completed.")
    print(f"   Training records: {train_df.count()}")
    print(f"   Testing records : {test_df.count()}\n")

    # ==================================================
    # 5️⃣ Build ML Pipeline
    # ==================================================
    print("🧠 Building Machine Learning pipeline...")

    tokenizer = Tokenizer(inputCol="post_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=2000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

    pipeline = Pipeline(stages=[
        tokenizer,
        remover,
        hashingTF,
        idf,
        lr
    ])

    print("✅ Pipeline created successfully.\n")

    # ==================================================
    # 6️⃣ Hyperparameter Tuning
    # ==================================================
    print("⚙️ Performing hyperparameter tuning with Cross Validation...")

    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [1000, 2000]) \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3
    )

    cvModel = cv.fit(train_df)

    print("✅ Model trained successfully using Cross Validation.\n")

    # ==================================================
    # 7️⃣ Generate Predictions
    # ==================================================
    print("📊 Generating predictions on test data...")

    predictions = cvModel.transform(test_df)

    # Convert numeric prediction back to original label
    label_converter = IndexToString(
        inputCol="prediction",
        outputCol="predictedLabel",
        labels=label_model.labels
    )

    predictions = label_converter.transform(predictions)

    print("✅ Predictions generated successfully.\n")

    # ==================================================
    # 8️⃣ Confusion Matrix (Proper Format)
    # ==================================================
    print("📈 Confusion Matrix (Actual vs Predicted)\n")

    confusion = predictions.groupBy(
        col("sentiment").alias("Actual"),
        col("predictedLabel").alias("Predicted")
    ).agg(count("*").alias("Count")) \
     .orderBy("Actual", "Predicted")

    confusion.show(truncate=False)

    print("✅ Confusion matrix generated successfully.\n")

    # ==================================================
    # 9️⃣ Model Evaluation
    # ==================================================
    print("📏 Evaluating model performance...")

    accuracy = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    ).evaluate(predictions)

    print(f"""
🎯 Model Performance
-----------------------
Accuracy  : {accuracy}
Precision : {precision}
Recall    : {recall}
F1 Score  : {f1}
""")

    print("✅ Model evaluation completed successfully.\n")

    # ==================================================
    # 🔟 Save Predictions to S3
    # ==================================================
    print("💾 Saving predictions to S3...")

    predictions.select(
        "post_id",
        "post_text",
        "sentiment",
        "predictedLabel"
    ).write.mode("overwrite").csv(
        "s3a://social-media-2026/sentiment_predictions_output/",
        header=True
    )

    print("✅ Predictions saved successfully to S3.\n")

    spark.stop()
    print("🛑 Spark session stopped successfully.")


if __name__ == "__main__":
    main()
