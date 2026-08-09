import numpy as np

path = "profiles\hp_webcam1_3meters.npz"   # change this

data = np.load(path, allow_pickle=True)

print("Keys in file:")
for key in data.files:
    arr = data[key]
    print(f"{key:15} shape={arr.shape} dtype={arr.dtype}")
    try:
        if arr.ndim == 0:
            print("   value =", arr.item())
    except:
        pass