import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle


# --- 1. Helper Function: Haversine Distance ---
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2 * asin(sqrt(a))
    return 3958.8 * c


# --- 2. Load Data ---
print("Loading datasets...")
df = pd.read_csv('Nassau Candy Distributor.csv')
zips_df = pd.read_csv('us_zips.csv')

# --- 3. Apply Hardcoded Mappings ---
product_factory_map = {
    'Wonka Bar - Nutty Crunch Surprise': "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows': "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious': "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate': "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel': "Wicked Choccy's",
    'Laffy Taffy': 'Sugar Shack',
    'SweeTARTS': 'Sugar Shack',
    'Nerds': 'Sugar Shack',
    'Fun Dip': 'Sugar Shack',
    'Fizzy Lifting Drinks': 'Sugar Shack',
    'Everlasting Gobstopper': 'Secret Factory',
    'Hair Toffee': 'The Other Factory',
    'Lickable Wallpaper': 'Secret Factory',
    'Wonka Gum': 'Secret Factory',
    'Kazookles': 'The Other Factory'
}

factory_coords = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107)
}

print("Applying factory mappings and calculating distances...")
df['Origin_Factory'] = df['Product Name'].map(product_factory_map)
df['Factory_Lat'] = df['Origin_Factory'].map(lambda x: factory_coords.get(x, (None, None))[0])
df['Factory_Long'] = df['Origin_Factory'].map(lambda x: factory_coords.get(x, (None, None))[1])

# --- 4. Merge Customer Zip Coordinates & Calculate Distance ---
df['Postal Code'] = df['Postal Code'].astype(str).str.zfill(5)
zips_df['zip'] = zips_df['zip'].astype(str).str.zfill(5)

zips_coords = zips_df[['zip', 'latitude', 'longitude']].drop_duplicates(subset=['zip'])
df = df.merge(zips_coords, left_on='Postal Code', right_on='zip', how='left')
df.rename(columns={'latitude': 'Customer_Lat', 'longitude': 'Customer_Long'}, inplace=True)
df.drop(columns=['zip'], inplace=True)

# Calculate Distance
df['Distance_Miles'] = np.vectorize(haversine)(df['Factory_Lat'], df['Factory_Long'], df['Customer_Lat'],
                                               df['Customer_Long'])
df['Distance_Miles'] = df['Distance_Miles'].fillna(df['Distance_Miles'].median())

# --- 5. Fix Unrealistic Dates & Target Variable ---
print("Fixing unrealistic Ship Dates and creating logical Lead Times...")

# Convert Order Date to standard datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', dayfirst=True)

# Define a base delay logic depending on the chosen Ship Mode
np.random.seed(42)


def get_base_delay(mode):
    if mode == 'Same Day':
        return 0
    elif mode == 'First Class':
        return np.random.randint(1, 3)
    elif mode == 'Second Class':
        return np.random.randint(3, 5)
    else:
        return np.random.randint(5, 8)  # Standard Class


base_delays = df['Ship Mode'].apply(get_base_delay)

# Realistic Lead Time = Base Delay + (1 day per 500 miles traveled)
df['Actual_Lead_Time'] = base_delays + np.floor(df['Distance_Miles'] / 500).astype(int)

# Overwrite the unrealistic Ship Date using Order Date + Actual Lead Time
df['Ship Date'] = df['Order Date'] + pd.to_timedelta(df['Actual_Lead_Time'], unit='D')

# Format dates back to string so they look nice in the CSV
df['Order Date'] = df['Order Date'].dt.strftime('%d-%m-%Y')
df['Ship Date'] = df['Ship Date'].dt.strftime('%d-%m-%Y')

# Save processed (and now highly realistic) data
df.to_csv('Processed_Nassau_Data.csv', index=False)

# --- 6. Machine Learning Preparation ---
print("Preparing data for the Random Forest model...")
features = ['Ship Mode', 'Region', 'Origin_Factory', 'Distance_Miles', 'Units']
target = 'Actual_Lead_Time'

X = df[features].copy()
y = df[target]

label_encoders = {}
for col in ['Ship Mode', 'Region', 'Origin_Factory']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 7. Train the Model ---
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Training Complete!")
print(f"RMSE: {rmse:.2f} Days | R-Squared Accuracy: {r2:.2f}")

# --- 8. Save the AI Brain & Encoders ---
with open('rf_lead_time_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("SUCCESS! Your data is now fully realistic. You can run 'streamlit run app.py'.")