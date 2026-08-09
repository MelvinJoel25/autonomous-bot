import json
import numpy as np

# input json file
json_file = "profiles\hp_webcam1_3meters.json"

# output npz file
npz_file = "profiles\hp_webcam1_3meters.npz"

with open(json_file, "r") as f:
    data = json.load(f)

converted = {}

for key, value in data.items():
    if isinstance(value, list):
        try:
            converted[key] = np.array(value, dtype=float)
        except:
            converted[key] = np.array(value, dtype=object)
    else:
        converted[key] = np.array(value)

np.savez_compressed(npz_file, **converted)

print("Saved:", npz_file)