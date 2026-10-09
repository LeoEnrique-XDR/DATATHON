from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).with_name("loopring_limpio.csv")
OUTPUT_CSV = Path(__file__).with_name("simulacion_loopring.csv")
OUTPUT_PLOT = Path(__file__).with_name("simulacion_loopring.png")
FORECAST_HORIZON = 7
TEST_FRACTION = 0.20


def main() -> None:
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if not {"date", "price"}.issubset(data.columns):
        raise ValueError("El CSV debe contener las columnas 'date' y 'price'.")

    prices = (
        data.assign(price=pd.to_numeric(data["price"], errors="coerce"))
        .dropna(subset=["date", "price"])
        .sort_values("date")
        .drop_duplicates(subset="date")
        .set_index("date")["price"]
    )
    if len(prices) < 20:
        raise ValueError("Se necesitan al menos 20 observaciones para la simulación.")

    first_test_index = int(len(prices) * (1 - TEST_FRACTION))
    if first_test_index < 2 or first_test_index >= len(prices):
        raise ValueError("No se pudo crear una división de entrenamiento y prueba válida.")

    records: list[dict[str, object]] = []
    for origin_index in range(first_test_index, len(prices), FORECAST_HORIZON):
        history = prices.iloc[:origin_index]
        actual_window = prices.iloc[
            origin_index : origin_index + FORECAST_HORIZON
        ]
        if actual_window.empty:
            continue

        steps = np.arange(len(history), dtype=float)
        coefficients = np.polyfit(steps, history.to_numpy(dtype=float), 1)
        future_steps = np.arange(
            len(history), len(history) + len(actual_window), dtype=float
        )
        trend_predictions = np.polyval(coefficients, future_steps)
        last_price_predictions = np.full(len(actual_window), history.iloc[-1])

        for offset, (date, actual) in enumerate(actual_window.items()):
            records.append(
                {
                    "forecast_origin": history.index[-1],
                    "date": date,
                    "actual_price": actual,
                    "last_price_prediction": last_price_predictions[offset],
                    "last_price_absolute_error": abs(
                        actual - last_price_predictions[offset]
                    ),
                    "linear_trend_prediction": trend_predictions[offset],
                    "linear_trend_absolute_error": abs(
                        actual - trend_predictions[offset]
                    ),
                }
            )

    simulation = pd.DataFrame(records)
    if simulation.empty:
        raise ValueError("No se generaron predicciones para el periodo de prueba.")

    simulation.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")

    last_price_mae = simulation["last_price_absolute_error"].mean()
    trend_mae = simulation["linear_trend_absolute_error"].mean()
    print("=== SIMULACIÓN HISTÓRICA DEL PRONÓSTICO DE LOOPRING ===")
    print(f"Periodo de entrenamiento inicial: {prices.index[0]:%Y-%m-%d} a "
          f"{prices.index[first_test_index - 1]:%Y-%m-%d}")
    print(f"Periodo simulado: {prices.index[first_test_index]:%Y-%m-%d} a "
          f"{prices.index[-1]:%Y-%m-%d}")
    print(f"Pronósticos evaluados: {len(simulation)}")
    print(f"MAE del último precio: {last_price_mae:.6f}")
    print(f"MAE de tendencia lineal: {trend_mae:.6f}")
    print(f"Resultados detallados: {OUTPUT_CSV.name}")

    plt.figure(figsize=(12, 6))
    plt.plot(
        prices.index[first_test_index:],
        prices.iloc[first_test_index:],
        color="black",
        label="Precio real (periodo simulado)",
    )
    plt.plot(
        simulation["date"],
        simulation["last_price_prediction"],
        color="darkorange",
        linestyle="--",
        marker="o",
        label=f"Último precio (MAE {last_price_mae:.4f})",
    )
    plt.plot(
        simulation["date"],
        simulation["linear_trend_prediction"],
        color="royalblue",
        linestyle=":",
        marker=".",
        label=f"Tendencia lineal (MAE {trend_mae:.4f})",
    )
    plt.title("Simulación histórica de pronósticos de Loopring (LRC)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    print(f"Gráfica de simulación: {OUTPUT_PLOT.name}")


if __name__ == "__main__":
    main()
