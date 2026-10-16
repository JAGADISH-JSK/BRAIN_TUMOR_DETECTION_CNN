import os, random, shutil
random.seed(42)
SRC, DST = "data", "data_small"
COUNTS = {"Training": 300, "Testing": 100}   # images per class

for split, n in COUNTS.items():
    for cls in os.listdir(f"{SRC}/{split}"):
        files = os.listdir(f"{SRC}/{split}/{cls}")
        os.makedirs(f"{DST}/{split}/{cls}", exist_ok=True)
        for f in random.sample(files, min(n, len(files))):
            shutil.copy(f"{SRC}/{split}/{cls}/{f}", f"{DST}/{split}/{cls}/{f}")
print("done")