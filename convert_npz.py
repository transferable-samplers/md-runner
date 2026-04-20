# convert_npz.py
import sys, os, numpy as np
from concurrent.futures import ProcessPoolExecutor

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)

def convert(name):
    src = os.path.join(SRC, name)
    dst = os.path.join(DST, name)
    if os.path.exists(dst):
        return
    with np.load(src) as z:
        # re-save uncompressed; same keys, same dtypes
        np.savez(dst, **{k: z[k] for k in z.files})
    print(name)

files = [f for f in os.listdir(SRC) if f.endswith(".npz")]
with ProcessPoolExecutor(max_workers=8) as ex:
    list(ex.map(convert, files))