# Databricks notebook source
# MAGIC %md
# MAGIC # Global Nike Retail Analytic Pipeline 

# COMMAND ----------

# MAGIC %md
# MAGIC ### # Project Overview
# MAGIC This Project processes learge-scale Nike retail catalog data from 45 countries using PySpark and Databricks. The Pipeline data ingestion, exploratory data analysis (EDA), data cleaning, transformation, business KPI generation, and Delta table creation for scalable retail analytics.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Ingestion
# MAGIC In this step, raw Nike retail CSV data is loaded into a PySpark DataFrame using Databricks

# COMMAND ----------

df = spark.read.csv(
    "/Volumes/workspace/default/nike_data/Global_Nike.csv",
    header=True,
    inferSchema=True
)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis (EDA)
# MAGIC This section explores the structure, schema, null values, column information, and overall qualitiy of the Nike retail dataset.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Check Schema

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Count Total Rows

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Column Names

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC Check Null Values

# COMMAND ----------

from pyspark.sql.functions import col, count, when
df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Total Countries

# COMMAND ----------

df.select("country_code").distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Important Columns

# COMMAND ----------

nike_df = df.select(
    "country_code",
    "product_name",
    "category",
    "subcategory",
    "currency",
    "price_local",
    "sale_price_local",
    "gender_segment",
    "availability_level",
    "available"
)

display(nike_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Cleaning
# MAGIC This step remove null values and duplicates records to improve overall data quality before transformation.

# COMMAND ----------

# MAGIC %md
# MAGIC Remove Null Product

# COMMAND ----------

nike_df = nike_df.dropna(subset=["product_name"])

# COMMAND ----------

# MAGIC %md
# MAGIC Remove Duplicates

# COMMAND ----------

nike_df = nike_df.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC Cleaned Row Count

# COMMAND ----------

nike_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Transformation
# MAGIC In this step, business-ready transformation and aggregation are performed using PySpark to generate meaningful retail insights.

# COMMAND ----------

# MAGIC %md
# MAGIC Country-Wise Product Count

# COMMAND ----------

country_product = nike_df.groupBy("country_code") \
    .count() \
        .orderBy("count", ascending=False)

display(country_product)

# COMMAND ----------

# MAGIC %md
# MAGIC Most Expensive Products

# COMMAND ----------

expensive_products = nike_df.orderBy(
    nike_df.price_local.desc()
)
display(expensive_products)

# COMMAND ----------

# MAGIC %md
# MAGIC Category Analysis

# COMMAND ----------

category_analysis = nike_df.groupBy("category") \
    .count() \
        .orderBy("count", ascending=False)

display(category_analysis)        

# COMMAND ----------

# MAGIC %md
# MAGIC ### Business KPI Analysis
# MAGIC This section generates retail business metrics such as country-wise product counts, category analysis, average pricing, and product availability insights.

# COMMAND ----------

# MAGIC %md
# MAGIC Average price by country

# COMMAND ----------

from pyspark.sql.functions import col, avg, when

nike_df_clean = nike_df.withColumn(
    "price_fixed", 
    when(
        col("price_local").rlike("^[0-9.]+$"),
        col("price_local").cast("double")
    )
)
avg_price_country = nike_df_clean.groupBy("country_code") \
    .agg(avg("price_fixed").alias("avg_price")) \
        .orderBy(col("avg_price").desc())

display(avg_price_country)        

# COMMAND ----------

# MAGIC %md
# MAGIC most common Product category

# COMMAND ----------

top_categories = nike_df.groupBy("category") \
    .count() \
        .orderBy("count", ascending=False)

display(top_categories)        

# COMMAND ----------

# MAGIC %md
# MAGIC Product Availability Analysis

# COMMAND ----------

availability_analysis = nike_df.groupBy("available") \
    .count()
display(availability_analysis)        

# COMMAND ----------

# MAGIC %md
# MAGIC Gender Segment Distribution

# COMMAND ----------

gender_analysis = nike_df.groupBy("gender_segment") \
    .count() \
        .orderBy("count", ascending=False)

display(gender_analysis)        

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retail Insights
# MAGIC Country-wise Average Product Price Analysis
# MAGIC Top Product Categories

# COMMAND ----------

from pyspark.sql.functions import count

top_categories = nike_df.groupBy("category") \
    .agg(count("*").alias("total_products")) \
        .orderBy(col("total_products").desc())

display(top_categories)        

# COMMAND ----------

# MAGIC %md
# MAGIC Countries with Highest Product Count

# COMMAND ----------

# MAGIC %md
# MAGIC Discount Analysis

# COMMAND ----------

from pyspark.sql.functions import col

discount_df = nike_df.filter(
    col("sale_price_local").isNotNull()
)
display(
    discount_df.select(
        "product_name",
        "country_code",
        "price_local",
        "sale_price_local" 
           ).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delta Table Creation
# MAGIC The cleaned and transformed dataset is stored as a Delta Table for scalable analysis and optimized querying

# COMMAND ----------

# MAGIC %md
# MAGIC Saving Cleaned Data into Delta Table

# COMMAND ----------

nike_df.write.format("delta") \
    .mode("overwrite") \
        .saveAsTable("nike_cleaned_data")

# COMMAND ----------

# MAGIC %md
# MAGIC Read Delta Table

# COMMAND ----------

delta_df = spark.read.table("nike_cleaned_data")
display(delta_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Partitioned Delta Table Creation
# MAGIC Partitioning the dataset by country_code to optimize query performance and scalable storage in Delta Lake.

# COMMAND ----------

nike_df.write.format("delta") \
    .mode("overwrite") \
        .partitionBy("country_code") \
            .saveAsTable("nike_partitioned_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Partitioned Table

# COMMAND ----------

partitioned_df = spark.read.table("nike_partitioned_data")
display(partitioned_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Window Function
# MAGIC Ranking Nike product within each country based on local pricing using PySpark window functions.

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, desc

window_spec = Window.partitionBy("country_code") \
    .orderBy(desc("price_local"))

ranked_df = nike_df.withColumn(
    "price_rank",
    row_number().over(window_spec)
)    

top_product_country = ranked_df.filter(
    ranked_df.price_rank <=3
)

display(
    top_product_country.select(
        "country_code",
        "product_name",
        "price_local",
        "price_rank"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Project Conclusion
# MAGIC This project demonstrates the implementation of a scalable big data retail analytics pipeline using PySpark and Databricks.
# MAGIC
# MAGIC The pipeline processes multi-country Nike retail catalog data containing over 1.4 million records across 45 countries. The workflow includes data ingestion, exploratory data analysis (EDA), data cleaning, transformation, business KPI generation, Delta Lake integration, partitioned storage optimization, and advanced window function analytics.
# MAGIC
# MAGIC Key technologies used:
# MAGIC - PySpark
# MAGIC - Databricks
# MAGIC - Delta Lake
# MAGIC - Distributed Data Processing
# MAGIC
# MAGIC This project showcases practical data engineering and retail analytics skills using modern big data technologies.
# MAGIC