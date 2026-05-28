import os
import pandas as pd

BASE_FOLDER = "/mnt/oldssd/aimotive-dataset/train/"
DATA_DIRS   = ["highway", "night", "rain", "urban"]
OUTPUT_CSV  = "./data/id_data.csv"

rows = []

for data_dir in DATA_DIRS:
    folder = os.path.join(BASE_FOLDER, data_dir)
    if not os.path.exists(folder):
        print(f"Nem létezik: {folder}")
        continue

    for section_id in os.listdir(folder):
        laz_dir = os.path.join(folder, section_id, "dynamic", "raw-revolutions")
        if not os.path.isdir(laz_dir):
            continue

        for fname in os.listdir(laz_dir):
            if fname.endswith(".laz"):
                frame_id = fname.replace("frame_", "").replace(".laz", "")
                rows.append({
                    "section_id": section_id,
                    "frame_id":   frame_id,
                })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV)
print(f"Összesen {len(df)} frame mentve → {OUTPUT_CSV}")
