import random
import pandas as pd
import numpy as np
import string as str
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as text
from scipy.interpolate import griddata

plt.style.use('classic')

dayOfBorn = pd.read_csv('C:\\Users\\Mielek\\Downloads\\births.csv')

scaleYear = dayOfBorn[dayOfBorn['year'] < 2001]

# scaleYear

fristSummer = scaleYear[(scaleYear['year'] >= 1969) & (scaleYear['year'] < 1990)]
secondSummer = scaleYear[(scaleYear['year'] >= 1990) & (scaleYear['year'] < 2001 )]


avarageFrist = fristSummer.groupby('month')['births'].mean()
avarageSecond = fristSummer.groupby('month')['births'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


plt.style.use('classic')

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Subplot 1: 1969–1989
ax1.scatter(avarageFrist.index, avarageFrist.values, color='steelblue', s=80, zorder=3)
ax1.set_title('Avarage births \n1969–1989', fontsize=13)
ax1.set_xlabel('Month')
ax1.set_ylabel('Avarege count born')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_labels, rotation=45)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.xaxis.set_major_formatter(plt.ScalarFormatter())
# Subplot 2: 1990–2000

ax2.scatter(avarageSecond.index, avarageSecond.values, color='tomato', s=80, zorder=3)
ax2.set_title('Average births\n1990–2000', fontsize=13)
ax2.set_xlabel('Month')
ax2.set_ylabel('Average count born')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_labels, rotation=45)
ax2.xaxis.set_major_formatter(plt.ScalarFormatter())
ax2.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Births in USA — comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
