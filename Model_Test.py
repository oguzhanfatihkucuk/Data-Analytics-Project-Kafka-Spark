import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

# Orijinal veriyi oluştur
data = {
    "Days_for_shipping_real": [3, 5, 4, 3, 2, 6, 2, 2, 3, 2, 6, 5, 4, 2, 2, 2, 5, 2, 0, 0,
                               5, 4, 3, 2, 6],
    "Days_for_shipment_scheduled": [4, 4, 4, 4, 4, 4, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1,
                                    0, 0, 4, 2, 2, 2, 2],
    "Benefit_per_order": [91.25, -249.09, -247.78, 22.86, 134.21, 18.58, 95.18, 68.43, 133.72, 132.15,
                          130.58, 45.69, 21.76, 24.58, 16.39, -259.58, -246.36, 23.84, 102.26,
                          87.18, 154.86, 82.30, 22.37, 17.70, 90.28],
    "Sales_per_customer": [314.64, 311.36, 309.72, 304.81, 298.25, 294.98, 288.42, 285.14, 278.59,
                           275.31, 272.03, 268.76, 262.20, 245.81, 327.75, 324.47, 321.20,
                           317.92, 314.64, 311.36, 309.72, 304.81, 298.25, 294.98, 288.42]
}

df = pd.DataFrame(data)

# Veri setini artırma
new_rows = []
for _ in range(1000):  # 100 yeni örnek ekle
    new_row = {
        "Days_for_shipping_real": np.random.randint(1, 7),
        "Days_for_shipment_scheduled": np.random.randint(0, 5),
        "Benefit_per_order": np.random.uniform(-300, 150)
    }
    new_rows.append(new_row)

# Yeni verileri DataFrame'e ekle
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Yeni bir özellik ekle
df['Total_Days'] = df['Days_for_shipping_real'] + df['Days_for_shipment_scheduled']

# NaN değerleri kontrol et
if df.isnull().values.any():
    print("NaN değerler bulundu. Eksik verileri kontrol ediyor...")
    # Eksik verileri doldur (örneğin, ortalama ile)
    df.fillna(df.mean(), inplace=True)

# Bağımsız ve bağımlı değişkenleri ayır
X = df[["Days_for_shipping_real", "Days_for_shipment_scheduled", "Benefit_per_order", "Total_Days"]]
y = df["Sales_per_customer"]

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Farklı modelleri tanımla
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor()
}

# Modelleri eğit ve değerlendir
for model_name, model in models.items():
    start_time = time.time()  # Zaman ölçmeye başla
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()  # Zamanı durdur

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    duration = end_time - start_time  # Süreyi hesapla

    print(f"{model_name}:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")
    print(f"  Süre: {duration:.4f} saniye\n")

# Hiperparametre ayarlaması için Gradient Boosting modeli
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_start_time = time.time()  # Zaman ölçmeye başla
grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
grid_end_time = time.time()  # Zamanı durdur

# En iyi model ve sonuçları
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

best_mse = mean_squared_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)
grid_duration = grid_end_time - grid_start_time  # Süreyi hesapla

print("Best Gradient Boosting Regressor:")
print(f"  Mean Squared Error: {best_mse:.2f}")
print(f"  R² Score: {best_r2:.2f}")
print(f"  Hyperparameter Tuning Time: {grid_duration:.4f} saniye\n")
