# TranSafe-Detecting-Anomalies-in-Transaction-Data
TranSafe: Detecting Anomalies in Transaction Data

Transaction Anomaly Detection

Overview

This project uses machine learning techniques to detect anomalies in financial transactions. The dataset consists of various transaction-related features, including transaction amount, volume, frequency, and customer demographics. The Isolation Forest algorithm is employed to identify anomalous transactions based on statistical deviations.

Features

The dataset includes the following key features:

Transaction_Amount: The amount of the transaction.

Transaction_Volume: The number of transactions in a given period.

Average_Transaction_Amount: The average transaction amount per user.

Frequency_of_Transactions: How often transactions occur.

Time_Since_Last_Transaction: Time elapsed since the last transaction.

Day_of_Week and Time_of_Day: Temporal attributes.

Age, Gender, Income: Customer demographic information.

Account_Type: Type of bank account used.

Installation and Dependencies

To run this project, you need Python and the following libraries:

pip install pandas plotly scikit-learn

Usage

Load the dataset: The data is read from a CSV file.

Exploratory Data Analysis (EDA): Various visualizations are used to understand data distributions and trends.

Anomaly Detection:

The mean and standard deviation of Transaction_Amount are calculated.

Transactions that exceed a threshold (mean + 2 std deviations) are flagged as anomalies.

An Isolation Forest model is trained and evaluated for more accurate anomaly detection.

User Input for Anomaly Prediction:

The user provides transaction values.

The trained model predicts whether the transaction is normal or anomalous.

Model Performance

The Isolation Forest model achieves perfect precision and recall in this dataset, detecting anomalies effectively.

Example Usage

Enter the value for 'Transaction_Amount': 6000
Enter the value for 'Average_Transaction_Amount': 700
Enter the value for 'Frequency_of_Transactions': 2
Anomaly detected: This transaction is flagged as an anomaly.

Visualization

The project generates visualizations including:

Distribution of transaction amounts.

Box plot of transaction amounts by account type.

Scatter plot of average transaction amount vs. age.

Bar chart showing transaction count by day of the week.

Scatter plot highlighting detected anomalies.

Conclusion

This project demonstrates an effective approach to detecting financial fraud using machine learning. It combines statistical analysis with an Isolation Forest model to identify unusual transactions, helping financial institutions prevent fraudulent activities.

Future Improvements

Incorporate more features like transaction location and merchant details.

Test additional anomaly detection techniques.

Deploy the model as a web application for real-time anomaly detection.

Author
Praveen Kumar B


