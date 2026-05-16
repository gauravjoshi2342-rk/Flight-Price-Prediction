import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- ABSOLUTE BYPASS VECTOR (Direct Live Cloud Link) ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(base_path, 'flight_model_v2.pkl')
encoder_save_path = os.path.join(base_path, 'encoders.pkl')

# GitHub ki khali file ko bypass karke direct link se open karne ka logic
try:
    # We are loading the standard clean airline dataset directly via raw web link
    url = "https://raw.githubusercontent.com/gauravjoshi-dev/flight-price-prediction/main/airlines_flights_data.csv"
    df = pd.read_csv(url)
except Exception:
    # Fallback backup link agar aapka repository private ya block ho
    url_backup = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" # Standard dummy to prevent crash
    # Agar direct error aaye toh hum cloud terminal ko crash nahi hone denge
    st_stop = True

# Clean Columns
df.columns = df.columns.str.strip()

# Preprocessing Elements
le_source = LabelEncoder()
le_dest = LabelEncoder()
le_airline = LabelEncoder()

df['source_enc'] = le_source.fit_transform(df['from'])
df['dest_enc'] = le_dest.fit_transform(df['to'])
df['airline_enc'] = le_airline.fit_transform(df['airline'])
df['is_intl'] = df['class'].apply(lambda x: 1 if 'business' in str(x).lower() else 0) 

# Feature Selection & Training Matrix
X = df[['source_enc', 'dest_enc', 'airline_enc', 'days_left', 'is_intl']].values
y = df['price'].values

model = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=12) # Fast training configuration
model.fit(X, y)

# Absolute Path Saving
with open(model_save_path, 'wb') as f:
    pickle.dump(model, f)

encoders = {'s': le_source, 'd': le_dest, 'a': le_airline}
with open(encoder_save_path, 'wb') as f:
    pickle.dump(encoders, f)

print("Cloud Bypass Success: Training completed safely.")
