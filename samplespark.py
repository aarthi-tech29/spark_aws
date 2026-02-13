from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
 
def main():
 
    spark = SparkSession.builder \
        .appName("Day39_40_Data_Cleaning_Project") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider") \
        .getOrCreate()
 
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session started")
 
    #  Use s3a://
    input_path = "s3a://spark-sales-2026/sales_raw.csv"
 
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    print("Raw data loaded")
 
    clean_df = (
        df.dropna()
          .withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd"))
          .filter(col("price") > 0)
    )
    print("Data cleaning completed")
 
    #  Use s3a://
    output_path = "s3a://spark-sales-2026/clean-data/sales_clean"
 
    clean_df.write.mode("overwrite").parquet(output_path)
    print("Clean data successfully written to S3 in Parquet format")
 
    spark.stop()
    print("Spark session stopped cleanly")
 
if __name__ == "__main__":
    main()