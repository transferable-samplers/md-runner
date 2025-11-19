import os

import numpy as np

path = "/network/scratch/a/alexander.tong/energy_temp/data/md/A_capped/"
for file in os.listdir(path):
    if not file.endswith("100000"):
        continue
    filelist = [npz_file for npz_file in os.listdir(os.path.join(path, file))]
    numbers = sorted([int(npz_file.split(".")[0]) for npz_file in filelist])
    numbers = numbers[-100:]
    if len(numbers) != 100:
        print("Missing files:", file, len(numbers))
        continue
    arrays = [
        np.load(os.path.join(path, file, str(number) + ".npz")) for number in numbers
    ]
    arr = np.concatenate([array["all_positions"] for array in arrays])
    np.savez(os.path.join(path, file + ".npz"), all_positions=arr)
    print(f"Saved file {os.path.join(path, file + '.npz')} with length {len(arr)}")
    print(file, arr.shape)
