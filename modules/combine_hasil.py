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
OUTPUT_DIR = os.path.join(BASE_DIR, "web_app", "static", "assets", "coastlines")
BASE_MODULES = os.path.dirname(__file__)

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
            # filepath = f"modules/LANDSAT8/Landsat8_Predict_{year}_{period}.tif"
            filepath = os.path.join(BASE_MODULES, 'LANDSAT8', f"Landsat8_Predict_{year}_{period}.tif")
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
                
    # ==== Path dasar & parameter ====
    years = range(2019, 2025)
    periods = ["q1", "q2", "q3", "q4"]
                
    for year in years:
        for period in periods:
            #filepath = f"modules/SENTINEL2/prediction_final_{year}_{period}.tif"
            filepath = os.path.join(BASE_MODULES, 'SENTINEL2', f"prediction_final_{year}_{period}.tif")
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
    years = sorted(set([c["year"] for c in coastlines_all]))

    # Colormap 1 warna
    cmap = cm.Blues  
    # Normalisasi tahun → range 0–1 untuk colormap
    norm = plt.Normalize(min(years), max(years))
    # Map tahun -> warna gradasi
    color_map = {y: cmap(norm(y)) for y in years}
    
    # --- Plot gabungan ---
    plt.figure(figsize=(12, 10))
    
    for item in coastlines_all:
        year = item["year"]
        period = item["period"]
        coastline = item["coastline"]

        for contour in coastline:
            xs = [pt[0] for pt in contour]  # longitude
            ys = [pt[1] for pt in contour]  # latitude
            plt.plot(xs, ys, color=color_map[year], linewidth=1,
                     label=f"{period} {year}")

    # Hilangkan duplikat legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("Gabungan Coastline All")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.savefig(os.path.join(OUTPUT_DIR, "coastline_combined_all.png"),
                dpi=300, bbox_inches='tight')

    return coastlines_all

def generate_coastline_compare(startYear, endYear):
    coastlines_all = generate_coastline_all()
    years = list(range(startYear, endYear + 1))
    periods = ["Jan_Jun", "Jul_Des", "q1", "q2", "q3", "q4"]

    # Base colormap 
    base_cmaps = [
        cm.Blues,
        cm.Reds,
        cm.Greens,
        cm.Purples,
        cm.Oranges,
        cm.Greys,
        cm.pink,
        cm.BuPu,
        cm.GnBu,
        cm.YlOrBr,
        cm.YlGn,
        cm.RdPu
    ]

    # Map tahun → colormap
    year_cmap_map = {}
    for i, y in enumerate(years):
        year_cmap_map[y] = base_cmaps[i]
    # Mapping warna final
    color_map = {}
    grad_positions = [0.55, 0.9, 0.3, 0.55, 0.75, 0.9]

    for y in years:
        cmap = year_cmap_map[y]
        for idx, p in enumerate(periods):
            color_map[(y, p)] = cmap(grad_positions[idx])

    # --- Plot gabungan ---
    plt.figure(figsize=(12, 10))

    for item in coastlines_all:
        year = item["year"]
        period = item["period"]

        if year not in years:
            continue
        coastline = item["coastline"]
        for contour in coastline:
            xs = [pt[0] for pt in contour]
            ys = [pt[1] for pt in contour]

            plt.plot(
                xs, ys,
                color=color_map[(year, period)],
                linewidth=1,
                label=f"{period} {year}"
            )

    # — Legend —
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(f"Coastline Comparison {startYear}–{endYear}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")

    out_path = os.path.join(OUTPUT_DIR, f"coastline_compare_{startYear}-{endYear}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

    return out_path

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
        plt.plot(xs, ys, color=colors[i], linewidth=1.5, label=f'Average {data["group_name"]}')

    plt.title(f"Average Coastlines {x} Lines")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="best", fontsize=9)
    plt.axis("equal")
    # plt.savefig(os.path.join(OUTPUT_DIR, f"coastline_combined_{x}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f"coastline_combined_{x}.png"),dpi=300, bbox_inches='tight')
    plt.show()

def generate_coastline_compare_avg(curYear, num_points=1000):
    coastlines_all = generate_coastline_all()
    year_groups = {
        f"{curYear}": [curYear],
        f"{curYear+1}": [curYear+1]
    }
    avg_coastlines = []

    for group_name, group_years in year_groups.items():
        coastlines = []
        for c in coastlines_all:
            if c["year"] in group_years:
                coastline_interp = interpolate_line(c["coastline"][0], num_points)
                coastlines.append(coastline_interp)
        mean_coastline = np.mean(coastlines, axis=0)
        avg_coastlines.append({
            "group_name": group_name,
            "mean_coastline": mean_coastline
        })

    colors = cm.plasma(np.linspace(0, 1, len(avg_coastlines)))
    plt.figure(figsize=(12, 10))

    for i, item in enumerate(avg_coastlines):
        group_name = item["group_name"]
        mean_coastline = item["mean_coastline"]

        xs = mean_coastline[:, 0]
        ys = mean_coastline[:, 1]
        plt.plot(xs, ys, color=colors[i], linewidth=2, label=f"{group_name}")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"Coastline Rata-rata Tahun {curYear}–{curYear+1}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    # plt.savefig(f"coastlines/coastline_avg_{curYear}-{curYear+1}.png", dpi=300, bbox_inches='tight')
    out_path = os.path.join(OUTPUT_DIR, f"coastline_avg_{curYear}-{curYear+1}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()