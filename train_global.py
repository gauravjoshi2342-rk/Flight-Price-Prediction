import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

base_path = os.path.dirname(os.path.abspath(__file__))
target_file = os.path.join(base_path, 'airlines_flights_data.csv')

if not os.path.exists(target_file):
    print(f"Error: {target_file} not found.")
    exit()

df = pd.read_csv(target_file)
df['Is_International'] = 0

def generate_robust_intl_data(n=8000):
    intl_cities = ['Dubai', 'London', 'New York', 'Singapore']
    domestic_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
    airlines = ['Emirates', 'Air India', 'Indigo', 'Vistara', 'Singapore Airlines', 'British Airways']
    
    routes = []
    for d in domestic_cities:
        for i in intl_cities:
            routes.append((d, i))
            routes.append((i, d))
            
    data = []
    for _ in range(n):
        r = routes[np.random.randint(0, len(routes))]
        al = np.random.choice(airlines)
        days = np.random.randint(1, 61)
        
        if 'Dubai' in r:
            base_p = 14000 if al == 'Indigo' else (28000 if al == 'Emirates' else 20000)
            scaler = 350
        elif 'Singapore' in r:
            base_p = 18000 if al == 'Indigo' else (35000 if al == 'Singapore Airlines' else 24000)
            scaler = 450
        elif 'London' in r:
            base_p = 32000 if al == 'Air India' else (55000 if al == 'British Airways' else 42000)
            scaler = 600
        else: # New York
            base_p = 55000 if al == 'Air India' else (85000 if al == 'Emirates' else 70000)
            scaler = 900
            
        price = base_p + (60 - days) * scaler + np.random.randint(-2000, 2000)
        data.append([r[0], r[1], al, days, 1, price])
        
    return pd.DataFrame(data, columns=['source_city', 'destination_city', 'airline', 'days_left', 'Is_International', 'price'])

intl_df = generate_robust_intl_data()
final_df = pd.concat([df, intl_df], ignore_index=True)

le_s = LabelEncoder().fit(final_df['source_city'])
le_d = LabelEncoder().fit(final_df['destination_city'])
le_a = LabelEncoder().fit(final_df['airline'])

final_df['source_city'] = le_s.transform(final_df['source_city'])
final_df['destination_city'] = le_d.transform(final_df['destination_city'])
final_df['airline'] = le_a.transform(final_df['airline'])

X = final_df[['source_city', 'destination_city', 'airline', 'days_left', 'Is_International']]
y = final_df['price']

model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1).fit(X, y)

pickle.dump(model, open(os.path.join(base_path, 'flight_model_v2.pkl'), 'wb'))
pickle.dump({'s': le_s, 'd': le_d, 'a': le_a}, open(os.path.join(base_path, 'encoders.pkl'), 'wb'))

print("Global Multi-Sector Training Execution Successful.")
