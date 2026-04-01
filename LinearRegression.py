from sklearn.manifold import Isomap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import numpy as np
import string as str
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


dataPopulation = pd.read_csv('')

countries = ["Argentina"]

years = np.arange(2025, 2030)

# plt.figure(figsize=(14, 7))

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
    aModel = model.fit(X_train, y_train).predict(X_test)
    bmodel = model.fit(X_test, y_test).predict(X_train)
    points = model.score(X_test, y_test)
    
    print(f"  R² na zbiorze testowym: {points:.4f}")
    print(f"  Współczynnik (nachylenie): {model.coef_[0]:,.0f} os/rok")
    print(f"  Wyraz wolny: {model.intercept_:,.0f}")

    # Years 2025-2029
    X_future = X_test.reshape(-1, 1)
    y_future = model.predict(X_future)  

    plt.plot(X_future, y_future,
             lw=2, linestyle="--", label=f"{country} (predykcja lin.)")

    all_years = np.arange(df_country["year"].min(), 2030).reshape(-1, 1)
    y_line = model.predict(all_years)
    plt.plot(all_years, y_line,
             lw=1, linestyle=":", alpha=0.5, label=f"{country} (linia regresji)")

# Split line
plt.axvline(x=2020, color="black", lw=1.5, linestyle=":", label="2020 — border prediction")

plt.autoscale()
plt.xlabel("Years")
plt.ylabel("Population")
plt.title("Example Regression Linear")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()
