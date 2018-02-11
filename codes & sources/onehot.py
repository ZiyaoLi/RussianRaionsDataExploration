import pandas as pd
import numpy as np
data = pd.read_csv("data_reg.csv")
cats = ["thermal_power_plant",
        "incineration",
        "oil_chemistry",
        "radiation",
        "railroad_terminal",
        "big_market",
        "nuclear_reactor"]
p = np.zeros(len(data))
for i in range(len(cats)):
    p += data[cats[i]] * (2 ** i)

data2 = data.drop(cats, axis=1)
data3 = pd.concat([data2, pd.get_dummies(p, drop_first=True)], axis=1)
data3.to_csv("onehot.csv", index=None)
