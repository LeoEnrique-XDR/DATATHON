import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("layer2.csv")

# Ver dimensiones
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

# Ver primeras filas
print("\nPrimeras 5 filas:")
print(df.head())

# Revisar estructura, valores nulos y filas duplicadas
print("\n=== INFORMACIÓN GENERAL ===")
df.info()

print("\n=== VALORES NULOS ===")
print(df.isnull().sum())

print("\n=== FILAS DUPLICADAS ===")
print("Cantidad:", df.duplicated().sum())

loopring_rows = df.loc[df["name"].astype(str).str.casefold() == "loopring"]
if loopring_rows.empty:
    raise ValueError("No se encontró la fila de Loopring en el dataset.")

date_columns = pd.to_datetime(df.columns, format="%m/%d/%y", errors="coerce")
is_date_column = date_columns.notna()
raw_prices = pd.to_numeric(
    loopring_rows.iloc[0, is_date_column], errors="coerce"
)
raw_prices.index = date_columns[is_date_column]
if raw_prices.index.has_duplicates:
    raise ValueError("Loopring tiene fechas duplicadas; revisa el dataset antes de limpiar.")
prices = raw_prices.dropna().sort_index()

if prices.empty:
    raise ValueError("Loopring no tiene precios válidos en las columnas de fecha.")

clean_loopring = prices.rename_axis("date").rename("price").reset_index()
clean_path = "loopring_limpio.csv"
clean_loopring.to_csv(clean_path, index=False, date_format="%Y-%m-%d")
print("\n=== LIMPIEZA DE LOOPRING ===")
print(f"Precios faltantes descartados: {raw_prices.isna().sum()}")
print(f"Fechas duplicadas: {raw_prices.index.duplicated().sum()}")
print(f"Filas conservadas: {len(clean_loopring)}")
print(f"Copia limpia guardada en: {clean_path}")

print("\n=== ANÁLISIS DE LOOPRING ===")
print("Fechas con precio:", len(prices))
print("Precio inicial:", prices.iloc[0])
print("Precio final:", prices.iloc[-1])
print("Variación porcentual:", f"{(prices.iloc[-1] / prices.iloc[0] - 1) * 100:.2f}%")
print("\nResumen estadístico:")
print(prices.describe())

daily_returns = prices.pct_change().dropna()
drawdown = prices / prices.cummax() - 1
max_drawdown_date = drawdown.idxmin()
peak_date_before_drawdown = prices.loc[:max_drawdown_date].idxmax()

print("\n=== RENDIMIENTO Y RIESGO HISTÓRICO ===")
print("Días con datos:", len(prices))
print("Días al alza:", int((daily_returns > 0).sum()))
print("Días a la baja:", int((daily_returns < 0).sum()))
print("Cambio diario promedio:", f"{daily_returns.mean():.2%}")
print("Volatilidad diaria (desviación estándar):", f"{daily_returns.std():.2%}")
print("Mejor día:", f"{daily_returns.max():.2%}", f"({daily_returns.idxmax():%Y-%m-%d})")
print("Peor día:", f"{daily_returns.min():.2%}", f"({daily_returns.idxmin():%Y-%m-%d})")
print(
    "Mayor caída desde un máximo:",
    f"{drawdown.min():.2%}",
    f"(máximo del {peak_date_before_drawdown:%Y-%m-%d} "
    f"al mínimo del {max_drawdown_date:%Y-%m-%d})",
)
print(
    "Máximo del periodo:",
    f"{prices.max():.6f}",
    f"({prices.idxmax():%Y-%m-%d})",
)
print(
    "Mínimo del periodo:",
    f"{prices.min():.6f}",
    f"({prices.idxmin():%Y-%m-%d})",
)

print("\n=== PRONÓSTICO BASE DE LOOPRING: PRÓXIMOS 7 DÍAS ===")
if len(prices) < 10:
    raise ValueError("Se necesitan al menos 10 precios para validar el pronóstico.")

split_index = int(len(prices) * 0.8)
training_prices = prices.iloc[:split_index]
test_prices = prices.iloc[split_index:]
training_values = training_prices.to_numpy(dtype=float)
test_values = test_prices.to_numpy(dtype=float)
training_steps = np.arange(len(training_values), dtype=float)
test_steps = np.arange(len(training_values), len(prices), dtype=float)

trend_coefficients = np.polyfit(training_steps, training_values, 1)
trend_test_predictions = np.polyval(trend_coefficients, test_steps)
naive_test_predictions = np.full(len(test_values), training_values[-1])
trend_mae = np.mean(np.abs(test_values - trend_test_predictions))
naive_mae = np.mean(np.abs(test_values - naive_test_predictions))

print(f"Validación cronológica: {len(training_prices)} días de entrenamiento, "
      f"{len(test_prices)} días de prueba")
print(f"Error absoluto medio - tendencia lineal: {trend_mae:.6f}")
print(f"Error absoluto medio - último precio: {naive_mae:.6f}")

forecast_dates = pd.date_range(
    start=prices.index[-1] + pd.Timedelta(days=1),
    periods=7,
    freq="D",
)
if naive_mae <= trend_mae:
    selected_model = "Último precio (referencia)"
    forecast_values = np.full(len(forecast_dates), prices.iloc[-1])
else:
    selected_model = "Tendencia lineal"
    full_series_steps = np.arange(len(prices), dtype=float)
    full_coefficients = np.polyfit(full_series_steps, prices.to_numpy(dtype=float), 1)
    forecast_steps = np.arange(len(prices), len(prices) + len(forecast_dates))
    forecast_values = np.polyval(full_coefficients, forecast_steps)

forecast = pd.Series(forecast_values, index=forecast_dates, name="predicted_price")
print(f"Modelo seleccionado por menor error de validación: {selected_model}")
print(forecast.rename_axis("date").to_string(float_format=lambda value: f"{value:.6f}"))

forecast_path = "loopring_pronostico_7_dias.csv"
forecast.rename_axis("date").reset_index().to_csv(
    forecast_path, index=False, date_format="%Y-%m-%d"
)
print(f"Pronóstico guardado en: {forecast_path}")

plt.figure(figsize=(12, 6))
plt.plot(prices.index, prices.values, color="purple", label="Precio histórico")
plt.plot(
    forecast.index,
    forecast.values,
    color="darkorange",
    linestyle="--",
    marker="o",
    label="Pronóstico (7 días)",
)
plt.axvline(prices.index[-1], color="gray", linestyle=":", alpha=0.7)
plt.title("Loopring (LRC): precio histórico y pronóstico base")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plot_path = "loopring_evolucion.png"
plt.savefig(plot_path)
plt.close()
print(f"\nGráfica guardada en: {plot_path}")

# Explorar problemas de calidad de datos sin modificar el CSV
print("\n=== EXPLORACIÓN Y DETECCIÓN DE PROBLEMAS ===")
date_labels = pd.to_datetime(df.columns[4:], format="%m/%d/%y", errors="coerce")
valid_date_columns = df.columns[4:][date_labels.notna()]
price_data = df.loc[:, valid_date_columns].apply(pd.to_numeric, errors="coerce")

print(f"Valores faltantes en precios: {price_data.isna().sum().sum()} "
      f"de {price_data.size} ({price_data.isna().sum().sum() / price_data.size:.2%})")
missing_by_asset = price_data.isna().sum(axis=1)
missing_by_asset = missing_by_asset[missing_by_asset > 0]
if missing_by_asset.empty:
    print("No hay precios faltantes por activo.")
else:
    print("Precios faltantes por activo:")
    for row_index, missing_count in missing_by_asset.items():
        asset_name = df.loc[row_index, "name"]
        print(f"  {asset_name}: {missing_count} de {len(valid_date_columns)} "
              f"({missing_count / len(valid_date_columns):.1%})")

numeric_prices = price_data.to_numpy(dtype=float)
infinite_count = np.isinf(numeric_prices).sum()
nonpositive_count = ((price_data <= 0) & price_data.notna()).sum().sum()
print(f"Precios infinitos: {infinite_count}")
print(f"Precios cero o negativos: {nonpositive_count}")

identifier_columns = ["id", "symbol", "name"]
for column in identifier_columns:
    print(f"Identificadores vacíos en '{column}': {df[column].isna().sum()}")
    print(f"Valores duplicados en '{column}': {df[column].duplicated().sum()}")

ordered_dates = pd.to_datetime(valid_date_columns, format="%m/%d/%y").sort_values()
date_gaps = ordered_dates.to_series().diff().dropna().dt.days
gap_count = int((date_gaps != 1).sum())
print(f"Fechas diarias ausentes entre la primera y última fecha: {gap_count}")

chronological_columns = valid_date_columns[
    np.argsort(pd.to_datetime(valid_date_columns, format="%m/%d/%y").values)
]
chronological_prices = df.loc[:, chronological_columns].apply(
    pd.to_numeric, errors="coerce"
)
daily_changes = chronological_prices.pct_change(axis=1, fill_method=None)
large_changes = daily_changes.stack().dropna()
large_changes = large_changes[large_changes.abs() > 0.20]
print("Variaciones de precio superiores al 20% entre días consecutivos:",
      len(large_changes))
for (row_index, date), change in large_changes.items():
    print(f"  {df.loc[row_index, 'name']} — {date}: {change:.1%}")
