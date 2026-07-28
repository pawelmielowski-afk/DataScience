import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Wczytaj dane - podaj właściwą ścieżkę do pliku CSV
dataPopulation = pd.read_csv() 

countries = ["Argentina"]
years = np.arange(2025, 2030)

plt.figure(figsize=(14, 7))

for country in countries:
    df_country = dataPopulation[dataPopulation["country"] == country][["year", "population"]].dropna()

    # Dane historyczne
    plt.plot(df_country["year"], df_country["population"],
              lw=2, label=f"{country} (dane)")

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1_000)}M")
    )

    # Regresja liniowa
    X_country = df_country[["year"]].values
    y_country = df_country["population"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_country, y_country, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)          

    print(f"  R² na zbiorze testowym: {r2:.4f}")
    print(f"  Współczynnik (nachylenie): {model.coef_[0]:,.0f} os/rok")
    print(f"  Wyraz wolny: {model.intercept_:,.0f}")

    # Prawdziwa predykcja na przyszłość: 2025-2029
    y_future = model.predict(years.reshape(-1, 1))
    plt.plot(years, y_future,
              lw=2, linestyle="--", label=f"{country} (predykcja lin.)")

    # Linia regresji na całym zakresie (dane historyczne + przyszłość)
    all_years = np.arange(df_country["year"].min(), 2030).reshape(-1, 1)
    y_line = model.predict(all_years)
    plt.plot(all_years, y_line,
              lw=1, linestyle=":", alpha=0.5, label=f"{country} (linia regresji)")

    # Linia oznaczająca koniec danych historycznych
    plt.axvline(x=df_country["year"].max(), color="black", lw=1.5,
                linestyle=":", label="koniec danych historycznych")

plt.autoscale()
plt.xlabel("Lata")
plt.ylabel("Populacja")
plt.title("Przykład regresji liniowej")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()
