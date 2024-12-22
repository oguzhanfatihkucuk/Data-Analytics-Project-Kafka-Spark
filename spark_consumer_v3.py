import os
import shutil
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Initialize a list to store predictions for analysis and visualization purposes
predictions_list = []

# Define the folder where output plots will be stored
output_folder = "plots"

# Check if the output folder exists; if yes, delete and recreate it
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Initialize global variables for the batch counter and the linear regression model
batch_counter = 0
lr_model = None

# Define a function to initialize and manage the Spark streaming session
def start_spark_streaming():
    global batch_counter, lr_model  # Use global variables for tracking state

    # Initialize a Spark session with Kafka support for streaming data
    spark = SparkSession.builder \
        .appName("KafkaSparkStreaming") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
        .getOrCreate()

    # Print Kafka server and topic configuration for debugging
    print("Kafka bootstrap servers:", spark.conf.get("kafka.bootstrap.servers", "localhost:9092"))
    print("Kafka topic:", "deneme2")

    # Define the schema for the incoming JSON data
    # This structure describes the expected fields and their data types in the Kafka messages
    sensor_schema = StructType([
        StructField("Type", StringType(), True),
        StructField("Days_for_shipping_real", IntegerType(), True),
        StructField("Days_for_shipment_scheduled", IntegerType(), True),
        StructField("Benefit_per_order", FloatType(), True),
        StructField("Sales_per_customer", FloatType(), True),
        StructField("Delivery_Status", StringType(), True),
        StructField("Late_delivery_risk", IntegerType(), True),
        StructField("Category_Id", IntegerType(), True),
        StructField("Category_Name", StringType(), True),
        StructField("Customer_City", StringType(), True),
        StructField("Customer_Country", StringType(), True),
        StructField("Customer_Email", StringType(), True),
        StructField("Customer_Fname", StringType(), True),
        StructField("Customer_Id", IntegerType(), True),
        StructField("Customer_Lname", StringType(), True),
        StructField("Customer_Password", StringType(), True),
        StructField("Customer_Segment", StringType(), True),
        StructField("Customer_State", StringType(), True),
        StructField("Customer_Street", StringType(), True),
        StructField("Customer_Zipcode", StringType(), True),
        StructField("Department_Id", IntegerType(), True),
        StructField("Department_Name", StringType(), True),
        StructField("Latitude", FloatType(), True),
        StructField("Longitude", FloatType(), True),
        StructField("Market", StringType(), True),
        StructField("Order_City", StringType(), True),
        StructField("Order_Country", StringType(), True),
        StructField("Order_Customer_Id", IntegerType(), True),
        StructField("Order_Date", DateType(), True),
        StructField("Order_Id", IntegerType(), True),
        StructField("Order_Item_Cardprod_Id", IntegerType(), True),
        StructField("Order_Item_Discount", FloatType(), True),
        StructField("Order_Item_Discount_Rate", FloatType(), True),
        StructField("Order_Item_Id", IntegerType(), True),
        StructField("Order_Item_Product_Price", FloatType(), True),
        StructField("Order_Item_Profit_Ratio", FloatType(), True),
        StructField("Order_Item_Quantity", IntegerType(), True),
        StructField("Sales", FloatType(), True),
        StructField("Order_Item_Total", FloatType(), True),
        StructField("Order_Profit_Per_Order", FloatType(), True),
        StructField("Order_Region", StringType(), True),
        StructField("Order_State", StringType(), True),
        StructField("Order_Status", StringType(), True),
        StructField("Order_Zipcode", StringType(), True),
        StructField("Product_Card_Id", IntegerType(), True),
        StructField("Product_Category_Id", IntegerType(), True),
        StructField("Product_Description", StringType(), True),
        StructField("Product_Image", StringType(), True),
        StructField("Product_Name", StringType(), True),
        StructField("Product_Price", FloatType(), True),
        StructField("Product_Status", StringType(), True),
        StructField("Shipping_Date", DateType(), True),
        StructField("Shipping_Mode", StringType(), True)
    ])

    # Read streaming data from Kafka
    # Configure the Kafka server and topic to subscribe to
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "deneme2") \
        .load()

    # Convert the "key" and "value" columns from binary to string for further processing
    df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

    # Parse the JSON data in the "value" column using the defined schema
    json_df = df.select(from_json(col("value").cast("string"), sensor_schema).alias("data")).select("data.*")

    # Create a feature vector from multiple columns using VectorAssembler
    assembler = VectorAssembler(
        inputCols=["Days_for_shipping_real", "Days_for_shipment_scheduled",
                   "Benefit_per_order"],  # Add columns contributing to the prediction
        outputCol="features"
    )

    # Define a linear regression model with the feature vector as input
    lr = LinearRegression(featuresCol="features", labelCol="Sales_per_customer")

    # Function to process each incoming data batch
    def model_prediction(batch_df, batch_id):
        global batch_counter, lr_model  # Access global variables
        if not batch_df.isEmpty():  # Proceed if the batch is not empty
            batch_counter += 1  # Increment the batch counter

            # Split the data into training (80%) and testing (20%) subsets
            train_df, test_df = batch_df.randomSplit([0.8, 0.2])
            train_df = assembler.transform(train_df)  # Transform training data for the model

            # Retrain the model every 10 batches
            if batch_counter % 10 == 0:
                print(f"Retraining the model at batch {batch_counter}")
                lr_model = lr.fit(train_df)

            # If a model exists, make predictions on the test dataset
            if lr_model is not None:
                test_df = assembler.transform(test_df)
                predictions = lr_model.transform(test_df)
                predictions_list.extend(predictions.select("prediction", "Sales_per_customer").collect())

                # Visualization and error calculation for the predictions
                if len(predictions_list) > 0:
                    predictions_df = pd.DataFrame(predictions_list, columns=["prediction", "Sales_per_customer"])
                    predictions_df['actual'] = predictions_df['Sales_per_customer']
                    predictions_df['difference'] = predictions_df['actual'] - predictions_df['prediction']
                    predictions_df['percentage_error'] = (predictions_df['difference'] / predictions_df['actual']) * 100

                    print(predictions_df[['Sales_per_customer', 'prediction', 'difference', 'percentage_error']])

                    plt.clf()  # Clear the previous plot

                    # Scatter plot with low error points highlighted
                    low_error_mask = (abs(predictions_df['difference']) / predictions_df['actual']) < 0.15
                    plt.scatter(predictions_df.loc[low_error_mask, 'Sales_per_customer'],
                                predictions_df.loc[low_error_mask, 'prediction'], color='orange', alpha=0.5)

                    plt.scatter(predictions_df.loc[~low_error_mask, 'Sales_per_customer'],
                                predictions_df.loc[~low_error_mask, 'prediction'], color='blue', alpha=0.5)

                    plt.title('Sales per Customer vs Prediction')
                    plt.xlabel('Sales per Customer')
                    plt.ylabel('Prediction')

                    # Save the plot to the output folder
                    plt.savefig(f'{output_folder}/plot_batch_{batch_id}.png')
                    plt.close()

    try:
        # Start the streaming query and apply batch processing
        print("Starting the stream...")
        query = json_df.writeStream \
            .foreachBatch(model_prediction) \
            .outputMode("append") \
            .start()

        query.awaitTermination()  # Wait for the streaming to finish

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        spark.stop()  # Ensure the Spark session is stopped after completion

# Run the Spark streaming process when the script is executed directly
if __name__ == '__main__':
    start_spark_streaming()
