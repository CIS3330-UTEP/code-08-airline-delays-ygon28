import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
#If any of this libraries is missing from your computer. Please install them using pip.

# Part 1
filename = './Flight_Delays_2018.csv'
df = pd.read_csv(filename)
df2 = df[['DEP_DELAY', 'ARR_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]
# print(df2.describe().round(2))

df3 = df.groupby(by='OP_CARRIER_NAME')[['ARR_DELAY','DEP_DELAY']].agg('mean').round(2)
# print(df3)

query = "OP_CARRIER_NAME == 'Allegiant Air'"
df4 = df.query(query)[['DEP_DELAY','ARR_DELAY','OP_CARRIER_NAME',]]
# print(df4)


# Part 2
# ARR_DELAY is the column name that should be used as dependent variable (Y).
y = df4["ARR_DELAY"]
x = df4["DEP_DELAY"]
x = sm.add_constant(x)

model = sm.OLS(y,x).fit()
print(model.summary())

# Part 3

a, b = np.polyfit(df4["DEP_DELAY"], df4["ARR_DELAY"], 1)

plt.scatter(df4["DEP_DELAY"], df4["ARR_DELAY"], color='purple')

plt.plot(df4["DEP_DELAY"], a*df4["DEP_DELAY"] + b)

plt.text(1,12,f"y = {b:.3f} + {a:.3f} + x", size = 12)

plt.xlabel("Departure Delay")
plt.ylabel("Arrival Delay")
plt.show()

# # # Part 4
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 0, ax=ax)
plt.show()