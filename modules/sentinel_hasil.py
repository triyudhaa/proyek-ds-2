import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
random.seed(42)
np.random.seed(42)

import coastline

# ==== Path dasar & parameter ====
years = range(2019, 2025)
periods = ["q1", "q2", "q3", "q4"]
            
# ==== LOOP untuk baca & koreksi semua file ====
coastlines_all = []
for year in years:
    for period in periods:
        filepath = f"SENTINEL2/prediction_final_{year}_{period}.tif"

        try:
            # --- ekstraksi ---
            ocean_mask, contours_pixel, contours_geo, meta, array = coastline.extract_coastline_from_geotiff(
                filepath,
                year,
                period,
                water_value=1,
                land_value=0,
                ws = 7
            )
            transform = meta["transform"]
            coastlines_all.append({
                "year": year,
                "period": period,
                "mask": array,
                "transform": transform,
                "coastline": contours_geo
            })

        except Exception as e:
            continue
            # print(f"Gagal baca {filepath}: {e}")

# --- Setup warna ---
years = sorted(set([c["year"] for c in coastlines_all]))
periods = ["q1", "q2", "q3", "q4"]
colors = cm.tab20(np.linspace(0, 1, len(years) * len(periods)))  # palet warna

# Mapping kombinasi (year, period) ke warna unik
color_map = {}
i = 0
for y in years:
    for p in periods:
        color_map[(y, p)] = colors[i]
        i += 1

# --- Plot gabungan ---
plt.figure(figsize=(12, 10))

def generate_coastline_sentinel():
    for item in coastlines_all:
        year = item["year"]
        period = item["period"]
        coastline = item["coastline"]

        for contour in coastline:
            xs = [pt[0] for pt in contour]  # x (longitude)
            ys = [pt[1] for pt in contour]  # y (latitude)
            plt.plot(xs, ys, color=color_map[(year, period)], linewidth=1,
                    label=f"{period} {year}")

    # Hilangkan duplikat di legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("Gabungan Coastline Sentinel2")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.savefig("coastline_combined_s2.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    return coastlines_all