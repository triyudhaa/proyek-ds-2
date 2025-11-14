import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import os
random.seed(42)
np.random.seed(42)

import coastline

# --- Tentukan path folder utama proyek ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # naik 1 level dari modules/
OUTPUT_DIR = os.path.join(BASE_DIR, "web_app", "coastlines")

# --- pastikan folder output ada ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

def init_result():
    coastlines_all = []
    # ==== Path dasar & parameter ====
    years = range(2013, 2019)
    periods = ["Jan_Jun", "Jul_Des"]
                
    # ==== LOOP untuk baca & koreksi semua file ====
    for year in years:
        for period in periods:
            filepath = f"LANDSAT8/Landsat8_Predict_{year}_{period}.tif"

            try:
                # --- ekstraksi ---
                contours, contours_geo, meta, array = coastline.extract_coastline_from_geotiff_landsat(
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
                
    # ==== Path dasar & parameter ====
    years = range(2019, 2025)
    periods = ["q1", "q2", "q3", "q4"]
                
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
    return coastlines_all

def generate_coastline_all():
    coastlines_all = init_result()
    
    # --- Setup warna ---
    years = sorted(set([c["year"] for c in coastlines_all]))
    # print(years)
    periods = ["Jan_Jun", "Jul_Des", "q1", "q2", "q3", "q4"]
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

    plt.title("Gabungan Coastline All")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.savefig(os.path.join(OUTPUT_DIR, "coastline_combined_all.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    
    return coastlines_all

def interpolate_line(line, num_points):
    """Interpolasi garis pantai agar punya jumlah titik seragam."""
    line = np.array(line)
    if len(line) < 2:
        return line

    # Hitung jarak kumulatif antar titik
    dist = np.cumsum(np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1)))
    dist = np.insert(dist, 0, 0)

    # Buat target jarak dengan jumlah titik tetap
    target_dist = np.linspace(0, dist[-1], num_points)

    # Interpolasi untuk x dan y
    x_interp = np.interp(target_dist, dist, line[:, 0])
    y_interp = np.interp(target_dist, dist, line[:, 1])
    return np.column_stack((x_interp, y_interp))

def avg_coastline(x, num_points=1000):
    coastlines_all = generate_coastline_all()
    increment = 12/x
    years = range(2013,2024)
    year_start = 2013
    year_end = 2024
    year_group = {}

    avg_coastlines = []

    while (year_start < year_end):
        now = int(year_start + increment - 1)
        year_group[f'{year_start}-{now}'] = [y for y in years if y >= year_start and y <= now]
        year_start = int(now + 1)

    for group_name, group_years in year_group.items():
        coastlines = []
        for c in coastlines_all:
            if(c["year"] in group_years):
                coastlines.append(interpolate_line(c["coastline"][0], num_points))
        mean_coastline = np.mean(coastlines, axis=0)
        avg_coastlines.append({
            "group_name": group_name,
            "mean_coastline": mean_coastline
        })
    
    # === Visualisasi ===
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(avg_coastlines)))

    for i, data in enumerate(avg_coastlines):
        xs, ys = data["mean_coastline"][:, 0], data["mean_coastline"][:, 1]
        plt.plot(xs, ys, color=colors[i], linewidth=1.5, label=f"Average {data["group_name"]}")

    plt.title(f"Average Coastlines {x} Lines")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="best", fontsize=9)
    plt.axis("equal")
    plt.savefig(os.path.join(OUTPUT_DIR, f"coastline_combined_{x}.png"), dpi=300, bbox_inches='tight')
    plt.show()