<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Transaction Anomaly Detection Project</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      margin: 20px;
    }
    pre {
      background-color: #f4f4f4;
      padding: 10px;
      overflow-x: auto;
    }
    code {
      font-family: Consolas, monospace;
    }
    ul, ol {
      margin-left: 20px;
    }
    a {
      text-decoration: none;
      color: #0366d6;
    }
    a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <h1>Transaction Anomaly Detection Project</h1>
  
  <h2>Anomaly Threshold</h2>
  <p>
    In this project, an anomaly is flagged if the <code>Transaction_Amount</code> exceeds the mean by at least 2 standard deviations.
    This criterion is designed to capture the roughly 2% of transactions that are significantly higher than normal.
  </p>
  
  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#model-training">Model Training</a></li>
    <li><a href="#deployment">Deployment (Optional)</a></li>
    <li><a href="#algorithms">Algorithms Used</a></li>
  </ul>
  
  <h2 id="introduction">Introduction</h2>
  <p>
    This project focuses on detecting anomalies within transaction data. Unusually high transaction amounts might signal fraudulent behavior or system errors.
    Using the Isolation Forest algorithm, the project automatically identifies these outlier transactions. In addition to model training,
    the notebook provides extensive data visualization and an interactive cell for testing new transaction inputs.
  </p>
  
  <h2 id="installation">Installation</h2>
  <ol>
    <li>Clone the repository:</li>
  </ol>
  <pre><code>git clone https://github.com/&lt;your-username&gt;/transaction-anomaly-detection.git
cd transaction-anomaly-detection</code></pre>
  <ol start="2">
    <li>Create and activate a virtual environment (optional but recommended):</li>
  </ol>
  <pre><code>python3 -m venv env
# On Linux/Mac:
source env/bin/activate
# On Windows:
env\Scripts\activate</code></pre>
  <ol start="3">
    <li>Install the required dependencies:</li>
  </ol>
  <pre><code>pip install -r requirements.txt</code></pre>
  <ol start="4">
    <li>Download the dataset:</li>
  </ol>
  <p>
    Place your dataset (for example, <code>transaction_anomalies_dataset.csv</code>) in a directory named <code>data</code> in the project root.
    Ensure the file follows the expected format and column structure.
  </p>
  
  <h2 id="usage">Usage</h2>
  <h3>Data Exploration &amp; Visualization</h3>
  <ol>
    <li>Open the Jupyter Notebook <code>TranSafe_Detecting_Anomalies_in_Transaction_Data.ipynb</code>.</li>
    <li>Run the cells sequentially to load the data, inspect its structure, and generate visualizations (scatter plots, box plots, and bar charts) that help understand key features such as:
      <ul>
        <li><code>Transaction_Amount</code></li>
        <li><code>Average_Transaction_Amount</code></li>
        <li><code>Frequency_of_Transactions</code></li>
        <li>Other customer attributes (e.g., Age, Account_Type)</li>
      </ul>
    </li>
  </ol>
  
  <h3>Model Training and Evaluation</h3>
  <ol>
    <li>The notebook selects the following features for anomaly detection:
      <code>Transaction_Amount</code>, <code>Average_Transaction_Amount</code>, and <code>Frequency_of_Transactions</code>.
    </li>
    <li>It splits the data into training and testing sets, trains an Isolation Forest model (with <code>contamination=0.02</code>), and evaluates the model using a classification report.</li>
  </ol>
  <pre><code># Example code snippet from the notebook:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Load the dataset
data = pd.read_csv("data/transaction_anomalies_dataset.csv")

# Calculate threshold for anomalies
mean_amount = data['Transaction_Amount'].mean()
std_amount = data['Transaction_Amount'].std()
anomaly_threshold = mean_amount + 2 * std_amount

# Flag anomalies
data['Is_Anomaly'] = data['Transaction_Amount'] > anomaly_threshold

# Prepare features and target
features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
X = data[features]
y = data['Is_Anomaly']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Isolation Forest model
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_train)</code></pre>
  
  <h3>Predicting Anomalies for New Transactions</h3>
  <ol>
    <li>The notebook includes a cell that prompts the user to input values for the selected features.</li>
    <li>These inputs are used to create a new DataFrame and the model predicts whether the transaction is anomalous (flagged if predicted as <code>-1</code> by the Isolation Forest, which is then mapped to <code>1</code> for an anomaly).</li>
  </ol>
  <pre><code># Example code for user input prediction:
user_inputs = []
for feature in features:
    value = float(input(f"Enter the value for '{feature}': "))
    user_inputs.append(value)
    
user_df = pd.DataFrame([user_inputs], columns=features)
user_pred = model.predict(user_df)
result = "Anomaly detected" if user_pred[0] == -1 else "No anomaly detected"
print(result)</code></pre>
  
  <h2 id="model-training">Model Training</h2>
  <ol>
    <li>The model is trained using the Isolation Forest algorithm, which isolates anomalies by randomly partitioning the feature space.</li>
    <li>The contamination parameter is set to 0.02 to reflect the assumption that 2% of the transactions are anomalous.</li>
  </ol>
  
  <h2 id="deployment">Deployment (Optional)</h2>
  <p>
    You can deploy the trained model using a FastAPI server. First, save the model using <code>pickle</code>:
  </p>
  <pre><code>import pickle

with open("isolation_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)
</code></pre>
  <ol start="2">
    <li>Create a file (e.g., <code>fast_api.py</code>) with the following content to serve predictions via an API:</li>
  </ol>
  <pre><code>from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the trained model
with open("isolation_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.post("/predict")
def predict_anomaly(transaction: dict):
    # Expecting a JSON payload with keys:
    # Transaction_Amount, Average_Transaction_Amount, Frequency_of_Transactions
    user_df = pd.DataFrame([transaction])
    prediction = model.predict(user_df)
    result = "Anomaly" if prediction[0] == -1 else "Normal"
    return {"prediction": result}
</code></pre>
  <ol start="3">
    <li>Run the FastAPI server from the terminal:</li>
  </ol>
  <pre><code>uvicorn fast_api:app --reload</code></pre>
  <ol start="4">
    <li>Open your web browser and navigate to <a href="http://localhost:8000">http://localhost:8000</a> to view the API documentation and test the endpoint.</li>
  </ol>
  
  <h2 id="algorithms">Algorithms Used</h2>
  <p>
    The project uses the Isolation Forest algorithm for anomaly detection. Unlike traditional clustering or statistical methods,
    Isolation Forest isolates outliers by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values.
    Anomalies are easier to isolate and, thus, require fewer splits. This makes the algorithm efficient and effective for high-dimensional data.
  </p>
  
</body>
</html>
