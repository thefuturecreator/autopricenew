import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
import sqlite3
import joblib
import os

app = Flask(__name__)

# File paths for the 60 CSV files and zip_cache
# TODO: Replace with actual file names (e.g., 1cl.csv, 2cl.csv, ..., z311.csv)
file_paths = [
    '1cl.csv', '2cl.csv',  # Add all 60 file names here
    # Example: 'abc1.csv', 'xyz2.csv', ..., 'z311.csv'
]  # Must have 60 files
zip_cache_path = 'zip_cache.csv'

# Load zip code distances
try:
    zip_distances = pd.read_csv(zip_cache_path)
except FileNotFoundError:
    print(f"Error: {zip_cache_path} not found in directory.")
    exit(1)

# Mock vehicle weights (replace with public dataset)
vehicle_weights = {
    'Car': 3000,
    'SUV': 4500,
    'Truck': 5000
}
specific_weights = {
    ('Honda', 'Civic Sedan'): 2800,
    ('Ford', 'F-150'): 4700,
    ('Toyota', 'Camry'): 3300
}

# Load and parse CSV file
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['vehicles'] = df['vehicles'].apply(lambda x: json.loads(x.replace('""', '"')))
        df['num_vehicles'] = df['vehicles'].apply(len)
        df['year'] = df['vehicles'].apply(lambda x: int(x[0]['year']) if x and 'year' in x[0] else 0)
        df['make'] = df['vehicles'].apply(lambda x: x[0]['make'] if x and 'make' in x[0] else '')
        df['model'] = df['vehicles'].apply(lambda x: x[0]['model'] if x and 'model' in x[0] else '')
        df['type'] = df['vehicles'].apply(lambda x: x[0]['type'] if x and 'type' in x[0] else '')
        df['vehicle_price'] = df['price_carrier_total'].astype(float) / df['num_vehicles']
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Get distance from zip_cache.csv
def get_distance(pickup_zip, dropoff_zip):
    try:
        row = zip_distances[
            (zip_distances['pickup_zip'] == pickup_zip) & 
            (zip_distances['dropoff_zip'] == dropoff_zip)
        ]
        if not row.empty:
            return row['distance_miles'].iloc[0]
        print(f"Warning: No distance found for {pickup_zip} to {dropoff_zip}. Using default 1000 miles.")
        return 1000  # Default distance
    except Exception as e:
        print(f"Error in get_distance: {e}")
        return 1000

# Preprocess data
def preprocess_data(df):
    df['distance'] = df.apply(lambda x: get_distance(x['pickup_zip'], x['dropoff_zip']), axis=1)
    df['month'] = pd.to_datetime(df['first_pickup_date']).dt.month
    df['weight'] = df.apply(
        lambda x: specific_weights.get((x['make'], x['model']), vehicle_weights.get(x['type'], 3000)), axis=1
    )
    df['is_large_vehicle'] = df['weight'] > 4000
    df['season_factor'] = df['month'].map({
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.2, 6: 1.0,
        7: 1.0, 8: 1.0, 9: 1.0, 10: 1.1, 11: 1.1, 12: 1.0
    })
    east_coast = ['NY', 'NJ', 'PA', 'MD', 'VA', 'NC', 'SC', 'GA']
    df['is_east_to_fl'] = (df['pickup_state_code'].isin(east_coast)) & (df['dropoff_state_code'] == 'FL')
    df['is_fl_to_east'] = (df['pickup_state_code'] == 'FL') & (df['dropoff_state_code'].isin(east_coast))
    df['season_factor'] = np.where(
        (df['is_east_to_fl'] & df['month'].isin([10, 11])), 1.3,
        np.where((df['is_fl_to_east'] & df['month'].isin([10, 11])), 0.8,
        np.where((df['is_east_to_fl'] & df['month'].isin([4, 5])), 0.8,
        np.where((df['is_fl_to_east'] & df['month'].isin([4, 5])), 1.3, df['season_factor'])))
    )
    df['vehicle_runs'] = df['vehicle_runs'].astype(int)
    return df

# Train model
def train_model(df):
    df = preprocess_data(df)
    features = ['distance', 'year', 'month', 'weight', 'vehicle_runs', 'num_vehicles']
    X = df[features]
    y = df['vehicle_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'pricing_model.pkl')
    return model, features

# Pricing function
def get_price(model, features, pickup_zip, dropoff_zip, year, make, model, type, first_pickup_date, vehicle_runs, num_vehicles):
    input_data = pd.DataFrame({
        'pickup_zip': [pickup_zip],
        'dropoff_zip': [dropoff_zip],
        'year': [year],
        'make': [make],
        'model': [model],
        'type': [type],
        'first_pickup_date': [first_pickup_date],
        'vehicle_runs': [1 if vehicle_runs else 0],
        'num_vehicles': [num_vehicles],
        'pickup_state_code': ['KS'],
        'dropoff_state_code': ['CA']
    })
    input_data = preprocess_data(input_data)
    X = input_data[features]
    base_price = model.predict(X)[0] * num_vehicles
    seasonal_price = base_price * input_data['season_factor'].iloc[0]
    surcharge = 0
    surcharge_details = {}
    if input_data['is_large_vehicle'].iloc[0]:
        surcharge += 200 * num_vehicles
        surcharge_details['large_vehicle'] = 200 * num_vehicles
    if input_data['distance'].iloc[0] > 2000:
        surcharge += 100 * num_vehicles
        surcharge_details['long_distance'] = 100 * num_vehicles
    if not input_data['vehicle_runs'].iloc[0]:
        surcharge += 150 * num_vehicles
        surcharge_details['non_running'] = 150 * num_vehicles
    total_price = seasonal_price + surcharge
    return {
        'base_price': round(base_price, 2),
        'seasonal_price': round(seasonal_price, 2),
        'surcharge': surcharge,
        'total_price': round(total_price, 2),
        'surcharge_details': surcharge_details
    }

# Process multiple files
def process_multiple_files(file_paths, chunksize=10000):
    conn = sqlite3.connect('shipping_data.db')
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            df = load_data(file_path)
            if df is not None:
                df = preprocess_data(df)
                df.to_sql('shipments', conn, if_exists='append', index=False)
    conn.close()
    print("Data processing complete.")
    return pd.read_sql('SELECT * FROM shipments', conn)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model = joblib.load('pricing_model.pkl')
    features = ['distance', 'year', 'month', 'weight', 'vehicle_runs', 'num_vehicles']
    pricing = get_price(
        model, features,
        pickup_zip=data['pickup_zip'],
        dropoff_zip=data['dropoff_zip'],
        year=int(data['year']),
        make=data['make'],
        model=data['model'],
        type=data['type'],
        first_pickup_date=data['date'],
        vehicle_runs=data['vehicle_runs'] == 'true',
        num_vehicles=int(data['num_vehicles'])
    )
    return jsonify(pricing)

# Process files and train model at startup
if __name__ == '__main__':
    print("Starting data processing...")
    df = process_multiple_files(file_paths)
    print("Training model...")
    model, features = train_model(df)
    print("Model trained. Starting Flask app...")
    app.run(debug=True)
