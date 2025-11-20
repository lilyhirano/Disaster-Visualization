import numpy as np


from load import run

disaster_list = ["hurricane-matthew", "midwest-flooding", "socal-fire"]
data = run(disaster_list)





disaster_to_idx = {
    "hurricane-matthew": 0,
    "midwest-flooding": 1,
    "socal-fire": 2
}

Y = []

for d in disaster_list:
    n = len(data[d]["labels"])
    label = disaster_to_idx[d]
    Y.append(np.full(n, label))

Y = np.concatenate(Y)

np.save("npy_images/disaster_types.npy", Y)