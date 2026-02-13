import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

# Number of rows
n = 5000

# Generate data
data = {
    'transaction_id': range(1, n+1),
    'transaction_time': [fake.date_time_between(start_date='-2M', end_date='now').strftime('%d-%m-%Y %H:%M') for _ in range(n)],
    'transaction_amount': np.round(np.random.uniform(10, 5000, n), 2),
    'merchant': np.random.choice(['Grocery', 'Fuel', 'Electronics', 'Travel', 'Food', 'Clothing'], n),
    'customer_age': np.random.randint(18, 80, n),
    'customer_region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'is_fraud': np.random.choice([0, 1], n, p=[0.9, 0.1])  # 10% fraud
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_creditcard_transactions.csv', index=False)

print("CSV with 5000 synthetic transactions created successfully!")
