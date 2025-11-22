import numpy as np


from load import run

disaster_list = ["midwest-flooding", "socal-fire"]
data = run(disaster_list)





disaster_to_idx = {
    "midwest-flooding": 0,
    "socal-fire": 1
}

Y = []

for d in disaster_list:
    n = len(data[d]["labels"])
    label = disaster_to_idx[d]
    Y.append(np.full(n, label))

Y = np.concatenate(Y)

np.save("npy_images/2_MERGED_labels.npy", Y)