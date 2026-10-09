import os
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge

# 1. DESCARGA Y CARGA DE DATOS
path = kagglehub.dataset_download("yousefsaeedian/layer2-blockchain-historical-data")
csv_file = os.path.join(path, "layer2.csv")
df = pd.read_csv(csv_file)

# 2. TRANSFORMACIÓN DE ANCHO A LARGO (MELT)
metadata_cols = ['id', 'symbol', 'name', 'platforms']
date_cols = [col for col in df.columns if col not in metadata_cols]

df_long = pd.melt(df, id_vars=metadata_cols, value_vars=date_cols, var_name='date', value_name='price')
df_long['date'] = pd.to_datetime(df_long['date'], format='%m/%d/%y')
df_long = df_long.sort_values(['id', 'date']).reset_index(drop=True)

# Imputación
df_long['price'] = df_long.groupby('id')['price'].ffill().bfill()

# 3. ANÁLISIS EXPLORATORIO (EDA)
# Retorno porcentual base
df_long['return_1d'] = df_long.groupby('id')['price'].pct_change()

# Mercado L2 general
df_long['market_return_mean'] = df_long.groupby('date')['return_1d'].transform('mean')

# Generar Gráfico EDA 1: Rendimiento Acumulado
df_long['cum_return'] = df_long.groupby('id')['return_1d'].transform(lambda x: (1 + x.fillna(0)).cumprod() - 1)

plt.figure(figsize=(10, 5))
for token in df_long['id'].unique()[:6]:
    token_data = df_long[df_long['id'] == token]
    plt.plot(token_data['date'], token_data['cum_return'] * 100, label=token)
plt.title("Rendimiento Acumulado (%) - Top Protocolos L2 (180 Días)", fontsize=12)
plt.xlabel("Fecha")
plt.ylabel("Retorno Acumulado (%)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("1_rendimiento_acumulado.png", dpi=300)
plt.close()

# Generar Gráfico EDA 2: Matriz de Correlación entre L2s
pivoted_returns = df_long.pivot(index='date', columns='symbol', values='return_1d')
corr_matrix = pivoted_returns.corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlación de Retornos')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.title("Matriz de Correlación Inter-Ecosistema Layer 2", fontsize=12)
plt.tight_layout()
plt.savefig("2_matriz_correlacion.png", dpi=300)
plt.close()

# 4. FEATURE ENGINEERING PARA MACHINE LEARNING
for lag in [1, 2, 3, 5, 7]:
    df_long[f'return_lag_{lag}'] = df_long.groupby('id')['return_1d'].shift(lag)

for window in [3, 7, 14]:
    df_long[f'roll_mean_ret_{window}'] = df_long.groupby('id')['return_1d'].transform(lambda x: x.shift(1).rolling(window).mean())
    df_long[f'roll_std_ret_{window}'] = df_long.groupby('id')['return_1d'].transform(lambda x: x.shift(1).rolling(window).std())

for span in [3, 7, 14]:
    df_long[f'ema_ret_{span}'] = df_long.groupby('id')['return_1d'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())

df_long['target_return'] = df_long.groupby('id')['return_1d'].shift(-1)
df_clean = df_long.dropna().reset_index(drop=True)

# 5. MODELADO Y VALIDACIÓN TEMPORAL
features = [col for col in df_clean.columns if 'lag' in col or 'roll' in col or 'ema' in col or col in ['return_1d', 'market_return_mean']]

dates = df_clean['date'].sort_values().unique()
split_date = dates[int(len(dates) * 0.8)]

train = df_clean[df_clean['date'] < split_date]
val = df_clean[df_clean['date'] >= split_date]

X_train, y_train = train[features], train['target_return']
X_val, y_val = val[features], val['target_return']

# Modelos
model_lgb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.015, max_depth=4, random_state=42, verbose=-1)
model_cat = CatBoostRegressor(iterations=200, learning_rate=0.015, depth=4, random_seed=42, verbose=0)
model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.015, max_depth=3, random_state=42)
model_ridge = Ridge(alpha=10.0, random_state=42)

model_lgb.fit(X_train, y_train)
model_cat.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
model_ridge.fit(X_train, y_train)

# Predicciones
p_lgb, p_cat = model_lgb.predict(X_val), model_cat.predict(X_val)
p_xgb, p_ridge = model_xgb.predict(X_val), model_ridge.predict(X_val)

preds_ensemble = (0.35 * p_lgb) + (0.35 * p_cat) + (0.20 * p_xgb) + (0.10 * p_ridge)

# Métricas
val_prices = val['price'].values
actual_future_price = val_prices * (1 + y_val.values)
pred_future_price = val_prices * (1 + preds_ensemble)
mae_dolares = np.mean(np.abs(actual_future_price - pred_future_price))

# 6. GRÁFICOS DE MODELADO
val_plot = val.copy()
val_plot['actual_price'] = actual_future_price
val_plot['pred_price'] = pred_future_price

# Gráfico 3: Real vs Predicho
token_df = val_plot[val_plot['id'] == 'matic-network']
plt.figure(figsize=(10, 5))
plt.plot(token_df['date'], token_df['actual_price'], label='Precio Real', color='#1f77b4', linewidth=2)
plt.plot(token_df['date'], token_df['pred_price'], label='Precio Predicho (Ensemble)', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title("Validación Temporal: Precio Real vs Predicho (matic-network)", fontsize=12)
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("3_real_vs_predicho.png", dpi=300)
plt.close()

# Gráfico 4: Feature Importance
imp_df = pd.DataFrame({'feature': features, 'importance': model_lgb.feature_importances_}).sort_values('importance', ascending=True)
plt.figure(figsize=(8, 5))
plt.barh(imp_df['feature'].tail(10), imp_df['importance'].tail(10), color='#2ca02c')
plt.title("Top 10 Drivers de Predicción (Importancia de Variables)", fontsize=12)
plt.xlabel("Cortes en Árboles de Decisión")
plt.tight_layout()
plt.savefig("4_feature_importance.png", dpi=300)
plt.close()

# Exportar Submission
latest_records = df_clean.sort_values('date').groupby('id').last().reset_index()
preds_sub = (0.35 * model_lgb.predict(latest_records[features])) + (0.35 * model_cat.predict(latest_records[features])) + (0.20 * model_xgb.predict(latest_records[features])) + (0.10 * model_ridge.predict(latest_records[features]))
submission = pd.DataFrame({'id': latest_records['id'], 'symbol': latest_records['symbol'], 'current_price': latest_records['price'], 'predicted_return': preds_sub, 'predicted_future_price': latest_records['price'] * (1 + preds_sub)})
submission.to_csv("submission.csv", index=False)

print(f"Procesamiento completado. MAE Final: ${mae_dolares:.4f} USD")