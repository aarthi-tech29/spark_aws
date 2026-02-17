from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, coalesce,
    month, dayofmonth, dayofweek,
    sum, countDistinct, year
)
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline


def main():

    spark = SparkSession.builder \
        .appName("Sales_Project_Final_Corrected") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("✅ Spark session started")

    input_path = "s3a://spark-sales-500-datas/sales.csv"

    df = spark.read.option("header", True) \
                   .option("inferSchema", True) \
                   .csv(input_path)

    print("✅ Raw data loaded")

    # =====================================================
    # ✅ FIX MIXED DATE FORMATS
    # =====================================================

    df = df.withColumn(
        "order_date",
        coalesce(
            to_date(col("order_date"), "dd-MM-yyyy"),
            to_date(col("order_date"), "M/d/yyyy"),
            to_date(col("order_date"), "MM/dd/yyyy")
        )
    )

    # =====================================================
    # DATA CLEANING
    # =====================================================

    clean_df = (
        df
        .filter(col("order_date").isNotNull())
        .filter((col("price") > 0) & (col("qty") > 0))
        .withColumn("month", month(col("order_date")))
        .withColumn("day", dayofmonth(col("order_date")))
        .withColumn("day_of_week", dayofweek(col("order_date")))
        .withColumn("revenue", col("price") * col("qty"))
        .fillna({
            "discount_pct": 0,
            "city": "Unknown",
            "payment_method": "Unknown",
            "order_status": "Unknown"
        })
    )

    total_records = clean_df.count()
    print(f"📦 Records after cleaning: {total_records}")

    if total_records == 0:
        print("❌ No valid data found.")
        spark.stop()
        return

    print("✅ Data cleaned successfully")

    # =====================================================
    # 📊 BUSINESS ANALYTICS
    # =====================================================

    print("\n📌 Top 5 Cities by Revenue")

    top_cities = clean_df.groupBy("city") \
        .agg(sum("revenue").alias("total_revenue")) \
        .orderBy(col("total_revenue").desc()) \
        .limit(5)

    top_cities.show()

    print("\n📌 Top Selling Category")

    clean_df.groupBy("category") \
        .agg(sum("qty").alias("total_quantity")) \
        .orderBy(col("total_quantity").desc()) \
        .show(1)

    print("\n📌 Repeat Customer Rate")

    customer_orders = clean_df.groupBy("customer_id") \
        .agg(countDistinct("order_id").alias("order_count"))

    repeat_customers = customer_orders.filter(col("order_count") > 1).count()
    total_customers = customer_orders.count()

    repeat_rate = (repeat_customers / total_customers) * 100 if total_customers > 0 else 0

    print(f"Repeat Customers: {repeat_customers}")
    print(f"Total Customers : {total_customers}")
    print(f"Repeat Customer Rate: {repeat_rate:.2f}%")

    print("\n📌 Monthly Revenue Trend")

    monthly_revenue = clean_df.groupBy(
        year("order_date").alias("year"),
        month("order_date").alias("month")
    ).agg(
        sum("revenue").alias("monthly_revenue")
    ).orderBy("year", "month")

    monthly_revenue.show()

    # =====================================================
    # 🤖 MACHINE LEARNING
    # =====================================================

    categorical_cols = ["city", "payment_method", "order_status"]
    numeric_cols = ["qty", "discount_pct", "month", "day", "day_of_week"]

    # Ensure enough distinct values
    for c in categorical_cols:
        if clean_df.select(c).distinct().count() < 2:
            print(f"⚠ Not enough distinct values in {c}. Skipping ML.")
            spark.stop()
            return

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
        for c in categorical_cols
    ]

    encoders = [
        OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec")
        for c in categorical_cols
    ]

    assembler_inputs = numeric_cols + [f"{c}_vec" for c in categorical_cols]

    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="price",
        numTrees=50,
        maxDepth=6,
        seed=42
    )

    pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

    train_df, test_df = clean_df.randomSplit([0.8, 0.2], seed=42)

    if train_df.count() == 0:
        print("⚠ Not enough training data.")
        spark.stop()
        return

    model = pipeline.fit(train_df)
    print("✅ Model training completed")

    predictions = model.transform(test_df)

    print("\n📌 Sample Predictions")
    predictions.select("qty", "price", "prediction").show(10)

    evaluator = RegressionEvaluator(
        labelCol="price",
        predictionCol="prediction",
        metricName="rmse"
    )

    print(f"\n📊 RMSE: {evaluator.evaluate(predictions)}")

    # =====================================================
    # 💾 SAVE RESULTS TO S3
    # =====================================================

    output_base = "s3a://spark-sales-500-datas/output/"

    top_cities.coalesce(1).write.mode("overwrite") \
        .option("header", True) \
        .csv(output_base + "top_cities")

    monthly_revenue.coalesce(1).write.mode("overwrite") \
        .option("header", True) \
        .csv(output_base + "monthly_revenue")

    predictions.select("qty", "price", "prediction") \
        .coalesce(1) \
        .write.mode("overwrite") \
        .option("header", True) \
        .csv(output_base + "ml_predictions")

    print("✅ All outputs saved to S3 successfully!")

    spark.stop()
    print("✅ Spark session stopped safely")


if __name__ == "__main__":
    main()
