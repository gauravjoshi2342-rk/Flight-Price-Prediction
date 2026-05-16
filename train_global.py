import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- CLOUD PATH CONFIGURATION ---
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'airlines_flights_data.csv')
model_save_path = os.path.join(base_path, 'flight_model_v2.pkl')
encoder_save_path = os.path.join(base_path, 'encoders.pkl')

# 1. Load Dataset Safely
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Critical CSV missing at: {data_path}")

df = pd.read_csv(data_path)

# Clean Columns if whitespace exists
df.columns = df.columns.str.strip()

# 2. Preprocessing Elements
le_source = LabelEncoder()
le_dest = LabelEncoder()
le_airline = LabelEncoder()

df['source_enc'] = le_source.fit_transform(df['from'])
df['dest_enc'] = le_dest.fit_transform(df['to'])
df['airline_enc'] = le_airline.fit_transform(df['airline'])

# Handle international binary indicator mapping
df['is_intl'] = df['class'].apply(lambda x: 1 if 'business' in str(x).lower() else 0) 

# 3. Feature Selection Matrix
X = df[['source_enc', 'dest_enc', 'airline_enc', 'days_left', 'is_intl']].values
y = df['price'].values

# 4. Model Training Pipeline
model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=15)
model.fit(X, y)

# 5. ABSOLUTE PATH OBJECT DUMPING (Cloud Lock)
with open(model_save_path, 'wb') as f:
    pickle.dump(model, f)

encoders = {'s': le_source, 'd': le_dest, 'a': le_airline}
with open(encoder_save_path, 'wb') as f:
    pickle.dump(encoders, f)

print("Backend Matrix Training Completed Successfully via Cloud Paths.")
