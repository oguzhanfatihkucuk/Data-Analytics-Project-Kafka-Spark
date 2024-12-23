
## Topics:
1. [Summary](#summary)
2. [Scope and Purpose of the Project](#scope-and-purpose-of-the-project)
    1. [Data Source and Scenario](#data-source-and-scenario)
    2. [Data Transfer and Streaming Framework](#data-transfer-and-streaming-framework)
    3. [Processing Pipeline](#processing-pipeline)
3. [Technologies Used in the Project](#technologies-used-in-the-project)
    1. [Kafka](#kafka)
    2. [Spark](#spark)
    3. [Python](#python)
4. [Description of the Dataset](#description-of-the-dataset)
5. [Flow Chart of the Project](#flow-chart-of-the-project)
6. [Data Analysis Methods and Visualization of Data Analysis](#data-analysis-methods-and-visualization-of-data-analysis)
7. [Model Training Processes](#model-training-processes)
    1. [Feature Assembly](#feature-assembly)
    2. [Model Initialization](#model-initialization)
    3. [Prediction Function](#prediction-function)
    4. [Collecting Results](#collecting-results)
8. [Outputs and Results of the Project](#outputs-and-results-of-the-project)

---

## 1. Summary

The data in this project was collected in a database using Apache Kafka and processed with Apache Spark Streaming. The project aims to create a forecasting model and analyze sales forecasts per customer.

---

## 2. Scope and Purpose of the Project

This project aims to conduct real-time analysis of orders and customer data from an e-commerce dataset. The primary objective is to manage data streams, make real-time predictions, and deliver analytical insights that can enhance performance.

### 2.1 Data Source and Scenario

The dataset will focus on order and customer details relevant to e-commerce. For dataset selection, options include using a known dataset or choosing a scenario from sources such as Kaggle competitions.

### 2.2 Data Transfer and Streaming Framework

Apache Kafka, a message distribution system, is employed to handle data streaming, though the specific approach may vary based on project requirements. Kafka streams the data to Apache Spark, which will act as a consumer, enabling real-time data analytics.

### 2.3 Processing Pipeline

Based on the selected data and scenario, the data will undergo specific pre-processing stages. Afterward, an appropriate machine learning pipeline (ML pipeline) will be established to ensure that the data is processed correctly and efficiently.

---

## 3. Technologies Used in the Project

### 3.1 Kafka (Version 2.11-1.1.0)

Apache Kafka is a distributed event-streaming platform designed to handle data transfer in real-time. In this project, Kafka plays a crucial role in ensuring that data generated by the e-commerce system (such as customer and order information) flows continuously and efficiently to downstream consumers like Apache Spark.

### 3.2 Spark (Version 3.5.3)

Apache Spark is a powerful, open-source distributed computing system that is optimized for large-scale data processing. In this project, Spark is the main data processing framework that consumes data from Kafka and performs both real-time processing and machine learning model training.

### 3.3 Python (Version 3.12)

Python is a versatile and widely used programming language in data science, known for its extensive libraries and ease of integration with platforms like Spark and Kafka. Python acts as the backbone for managing, processing, and modeling data in this project.

---

## 4. Description of the Dataset

This dataset is a comprehensive collection of sales, customer, product, and delivery information for an e-commerce or retail business. With a total of 180,519 rows and 53 columns, each row in this dataset represents information related to a specific order. The dataset covers various dimensions, including customer demographics, order details, logistics information, and product attributes.

For reference, the dataset can be found here: **[Kaggle Dataset - Smart Supply Chain](https://www.kaggle.com/datasets/alinoranianesfahani/dataco-smart-supply-chain-for-big-data-analysis)**.

### Below is a detailed description of the dataset:
- **Order Id, Order Date, Shipping Date:** Provides information about order dates, shipment dates, and delivery times.
- **Days for shipping (real), Days for shipment (scheduled):** Planned and actual shipping durations, useful for evaluating delivery performance.
- **Product Name, Product Price, Product Category:** Specifies the product name, price, and category.
- **Customer Id, Customer Name, Customer Email, Customer Segment:** Captures customer identification, contact, and segment information.
- **Customer City, Customer Country:** Indicates the customer's geographical location.
- **Benefit per order, Sales per customer:** Reflects the financial contribution of each order or customer to the business.

This dataset provides a rich foundation for analyzing e-commerce metrics, customer behavior, and logistics performance.

---

## 5. Flow Chart of the Project

The flowchart below illustrates a data processing pipeline involving CSV data, Apache Kafka, and Apache Spark. Initially, data from CSV files is streamed into Apache Kafka, which serves as the messaging system. A broker manages the data distribution, enabling Kafka to act as the data source for Apache Spark. Apache Spark then consumes and processes this data, transforming it as needed. Finally, the processed data is outputted for further analysis and visualization using Python's matplotlib library (indicated as "plt" in the chart).
<img src="./png_files/Flowchart-1.png" alt="AppLoading.png" width="100%" height="auto" style="margin: 10px;"/>
**Flowchart-1**

---

## 6. Data Analysis Methods and Visualization of Data Analysis

In this project, the `matplotlib.pyplot` library (`plt`) is essential for visualizing the model’s performance, particularly through scatter plots that compare predicted values with actual customer sales. These plots allow us to see each prediction relative to actual values, offering a clear view of the model’s accuracy and behavior.

A central feature of these visualizations is the reference line `F(x) = x`, which represents perfect predictions. This line serves as a quick benchmark, where any data points deviating from it indicate prediction errors, highlighting areas for potential model improvement. Transparency adjustments (using alpha) make it easier to interpret overlapping points, especially in data-dense regions.

The project includes labels, titles, and legends to make each visualization more understandable. By saving each plot with unique identifiers, the project retains a visual record of performance across different data batches, allowing for model tracking and insights over time.

---

## 7. Model Training Processes

### 7.1 Feature Assembly:
The following columns are combined to create a new column called **features**:
- `Days_for_shipping_real`: Actual shipping durations.
- `Days_for_shipment_scheduled`: Scheduled shipping durations.
- `Benefit_per_order`: Profit generated per order.
- `Order_Item_Quantity`: Quantity of items in the order.
- `Order_Item_Discount`: Discounts applied to items in the order.
- `Product_Price`: Prices of the products.

### 7.2 Model Initialization

A linear regression model is instantiated with the specified feature and label columns. The `featuresCol` parameter is set to "features", while the `labelCol` is set to "Sales_per_customer", indicating the target variable to be predicted.

The decision to choose the linear regression model was made after training a small part of our data with various models (Model Test-1). The model that gave the most accurate results in the shortest time was selected.
<img src="./png_files/Model Test-1.png" alt="AppLoading.png" width="100%" height="auto" style="margin: 10px;"/>
**Model Test-1**

### 7.3 Prediction Function

The `model_prediction` function processes incoming data batches as follows:
1. **Data Handling:** Checks if the incoming DataFrame is not empty. If it contains data, it splits the data into training (80%) and test (20%) sets.
2. **Training the Model:** The training dataset is transformed to create the feature vector. If the model has not been trained yet, the model is fitted to this training data.
3. **Making Predictions:** After the model is trained, the test dataset is transformed, and predictions are made.
<img src="./png_files/Code Snippet-1.png" alt="AppLoading.png" width="200" style="margin: 10px;"/>
**Code Snippet-1**  
<img src="./png_files/Code Snippet-2.png" alt="AppLoading.png" width="100%" height="auto" style="margin: 10px;"/>
**Code Snippet-2**

### 7.4 Collecting Results

Predictions are collected along with the actual sales figures into a list. At this stage, necessary metrics to evaluate the model's performance are computed.
<img src="./png_files/Code Snippet-3.png" alt="AppLoading.png" width="100%" height="auto" style="margin: 10px;"/>
**Code Snippet-3**

---

## 8. Outputs and Results of the Project

The graphs below show the outputs of the project. The feature that is predicted in our project is **Sales_per_customer**, and the prediction data generated by our model for each row is labeled as **prediction**. In the graph below, `f(x) = x` was created to determine the relationship between our predictions and the real data. The closer each point on the chart is to this line, the more accurate our predictions are.

In our project, the error rate was chosen as 15 percent. Accordingly, the lines `f(x)=0.85` and `f(x)=1.15x` were drawn to determine the error rate between our predictions and the actual data. The data falling between these two lines are considered correctly predicted in our model and are marked with an orange label.
<img src="./png_files/Graph-1.png" alt="AppLoading.png" width="100%" height="auto" style="margin: 10px;"/>
**Graph-1**
