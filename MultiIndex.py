import pandas as pd
import numpy as np


index = [('California', 2010), ('California', 2020), ('New York', 2010), ('New York', 2020),  ('Floryda', 2010), ('Floryda', 2020) ]
population = [21804000, 23086000, 24973000, 37732000, 18908600, 23086000]

pop = pd.Series(population, index=index)

index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)

pd.Series(pop['California', 2020] - pop['New York', 2010], index=['1']).to_json()

index = pd.MultiIndex.from_product(
    [['USA','Canada'], ['2020','2021']],
    names=['Country','Year']
)

df = pd.DataFrame({'GDP':[21,22,1.6,1.7]}, index=index)
print(df)
