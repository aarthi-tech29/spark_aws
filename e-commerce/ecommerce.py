from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def main():

    # ==================================================
    # 1️⃣ Start Spark Session
    # ==================================================
    print("\n🚀 Starting Spark Session...\n")

    spark = SparkSession.builder \
        .appName("E-Commerce Recommendation System - ALS") \
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
        "s3a://ecommerce-2026/ecommerce_recommendation_50_records.csv",
        header=True,
        inferSchema=True
    ).dropna()

    print(f"✅ Dataset loaded successfully. Total records: {df.count()}\n")

    df.show(5)

    # ==================================================
    # 3️⃣ Train-Test Split
    # ==================================================
    print("🔀 Splitting dataset into training and testing sets...")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print(f"   Training records: {train_df.count()}")
    print(f"   Testing records : {test_df.count()}")
    print("✅ Data split completed successfully.\n")

    # ==================================================
    # 4️⃣ Build ALS Model
    # ==================================================
    print("🧠 Building ALS Recommendation Model...")

    als = ALS(
        userCol="user_id",
        itemCol="product_id",
        ratingCol="rating",
        rank=10,
        maxIter=10,
        regParam=0.1,
        coldStartStrategy="drop",  # Prevent NaN errors
        nonnegative=True
    )

    model = als.fit(train_df)

    print("✅ Model trained successfully using ALS.\n")

    # ==================================================
    # 5️⃣ Generate Predictions
    # ==================================================
    print("📊 Generating rating predictions on test data...")

    predictions = model.transform(test_df)

    predictions = predictions.withColumn(
        "prediction",
        round(col("prediction"), 2)
    )

    predictions.show()

    print("✅ Predictions generated successfully.\n")

    # ==================================================
    # 6️⃣ Confusion Matrix (Rating Based)
    # ==================================================
    print("📈 Creating Rating Comparison Table...\n")

    comparison = predictions.groupBy(
        col("rating").alias("Actual_Rating"),
        col("prediction").alias("Predicted_Rating")
    ).agg(count("*").alias("Count")) \
     .orderBy("Actual_Rating", "Predicted_Rating")

    comparison.show(truncate=False)

    print("✅ Rating comparison table generated successfully.\n")

    # ==================================================
    # 7️⃣ Model Evaluation
    # ==================================================
    print("📏 Evaluating model performance...")

    evaluator_rmse = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    evaluator_mae = RegressionEvaluator(
        metricName="mae",
        labelCol="rating",
        predictionCol="prediction"
    )

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    print(f"""
🎯 Model Performance
-----------------------
RMSE : {rmse}
MAE  : {mae}
""")

    print("✅ Model evaluation completed successfully.\n")

    # ==================================================
    # 8️⃣ Generate Top 3 Recommendations Per User
    # ==================================================
    print("🛍 Generating Top 3 product recommendations per user...")

    user_recommendations = model.recommendForAllUsers(3)

    user_recommendations.show(truncate=False)

    print("✅ Recommendations generated successfully.\n")

    # ==================================================
    # 9️⃣ Save Predictions to S3
    # ==================================================
    print("💾 Saving predictions to S3...")

    predictions.write.mode("overwrite").csv(
        "s3a://ecommerce-2026/recommendation_predictions_output/",
        header=True
    )

    print("✅ Predictions saved successfully to S3.\n")

    spark.stop()
    print("🛑 Spark session stopped successfully.")


if __name__ == "__main__":
    main()
