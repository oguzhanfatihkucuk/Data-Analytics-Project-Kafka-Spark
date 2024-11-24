from kafka import KafkaProducer
from flask import Flask
import pandas as pd
import json
import threading
import time

# Kafka Producer settings
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # Kafka broker address
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # Serialize data to JSON format
)

# Initialize Flask application
app = Flask(__name__)

# Function to read data from a CSV file and send it to Kafka
def produce_csv_data():
    # Read local CSV file (e.g., 'data.csv')
    data = pd.read_csv('C:/Users/90533/Desktop/data.csv', encoding='ISO-8859-1')

    # Send each row of data to Kafka
    for _, row in data.iterrows():
        # Convert the row to a dictionary
        order_data = {
            'Type': row['Type'],
            'Days_for_shipping_real': row['Days for shipping (real)'],
            'Days_for_shipment_scheduled': row['Days for shipment (scheduled)'],
            'Benefit_per_order': row['Benefit per order'],
            'Sales_per_customer': row['Sales per customer'],
            'Delivery_Status': row['Delivery Status'],
            'Late_delivery_risk': row['Late_delivery_risk'],
            'Category_Id': row['Category Id'],
            'Category_Name': row['Category Name'],
            'Customer_City': row['Customer City'],
            'Customer_Country': row['Customer Country'],
            'Customer_Email': row['Customer Email'],
            'Customer_Fname': row['Customer Fname'],
            'Customer_Id': row['Customer Id'],
            'Customer_Lname': row['Customer Lname'],
            'Customer_Password': row['Customer Password'],
            'Customer_Segment': row['Customer Segment'],
            'Customer_State': row['Customer State'],
            'Customer_Street': row['Customer Street'],
            'Customer_Zipcode': row['Customer Zipcode'],
            'Department_Id': row['Department Id'],
            'Department_Name': row['Department Name'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Market': row['Market'],
            'Order_City': row['Order City'],
            'Order_Country': row['Order Country'],
            'Order_Customer_Id': row['Order Customer Id'],
            'Order_Date': row['order date (DateOrders)'],
            'Order_Id': row['Order Id'],
            'Order_Item_Cardprod_Id': row['Order Item Cardprod Id'],
            'Order_Item_Discount': row['Order Item Discount'],
            'Order_Item_Discount_Rate': row['Order Item Discount Rate'],
            'Order_Item_Id': row['Order Item Id'],
            'Order_Item_Product_Price': row['Order Item Product Price'],
            'Order_Item_Profit_Ratio': row['Order Item Profit Ratio'],
            'Order_Item_Quantity': row['Order Item Quantity'],
            'Sales': row['Sales'],
            'Order_Item_Total': row['Order Item Total'],
            'Order_Profit_Per_Order': row['Order Profit Per Order'],
            'Order_Region': row['Order Region'],
            'Order_State': row['Order State'],
            'Order_Status': row['Order Status'],
            'Order_Zipcode': row['Order Zipcode'],
            'Product_Card_Id': row['Product Card Id'],
            'Product_Category_Id': row['Product Category Id'],
            'Product_Description': row['Product Description'],
            'Product_Image': row['Product Image'],
            'Product_Name': row['Product Name'],
            'Product_Price': row['Product Price'],
            'Product_Status': row['Product Status'],
            'Shipping_Date': row['shipping date (DateOrders)'],
            'Shipping_Mode': row['Shipping Mode']
        }

        # Send data to Kafka
        try:
            producer.send('deneme2', order_data)  # Send to Kafka topic 'deneme2'
            print(f"Data sent: {order_data}")  # Log the sent data
        except Exception as e:
            print(f"Message sending error: {e}")  # Log any errors in sending
        # time.sleep(0.01)  # Optional delay between sends

if __name__ == '__main__':
    # Run the producer in a separate thread
    threading.Thread(target=produce_csv_data, daemon=True).start()
    # Start the Flask app
    app.run(debug=True, port=5000)  # Flask app running on port 5000
