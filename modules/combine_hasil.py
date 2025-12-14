import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import os
import io
import math
from PIL import Image
random.seed(42)
np.random.seed(42)

from . import coastline

# --- Tentukan path folder utama proyek ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # naik 1 level dari modules/
OUTPUT_DIR = os.path.join(BASE_DIR, "web_app", "static", "assets", "coastlines")
OUTPUT_DIR_2 = os.path.join(BASE_DIR, "web_app", "static", "assets", "predictions")
BASE_MODULES = os.path.dirname(__file__)

# --- pastikan folder output ada ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_q_to_text(period):
        mapping = {
        "q1": "Jan_Mar",
        "q2": "Apr_Jun",
        "q3": "Jul_Sep",
        "q4": "Okt_Des"
        }
        return mapping.get(period, period)

def init_result():
    coastlines_all = []
    # ==== Path dasar & parameter ====
    years = range(2013, 2019)
    periods = ["Jan_Jun", "Jul_Des"]
    listPlot = []
                
    # ==== LOOP untuk baca & koreksi semua file ====
    for year in years:
        numPlot = 0
        for period in periods:
            # filepath = f"modules/LANDSAT8/Landsat8_Predict_{year}_{period}.tif"
            filepath = os.path.join(BASE_MODULES, 'LANDSAT8', f"Landsat8_Predict_{year}_{period}.tif")
            try:
                # --- ekstraksi ---
                ocean_mask, contours_pixel, contours_geo, meta, array, fig = coastline.extract_coastline_from_geotiff(
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
                    "group_name": f"{year} {period}",
                    "mask": array,
                    "transform": transform,
                    "coastline": contours_geo,
                    "plot": fig
                })
                numPlot+=1
                plt.close(fig)
            except Exception as e:
                continue
                # print(f"Gagal baca {filepath}: {e}")
        listPlot.append(numPlot)
        # print(listPlot)     

    # ==== Path dasar & parameter ====
    years = range(2019, 2025)
    periods = ["q1", "q2", "q3", "q4"]    
    for year in years:
        numPlot = 0
        for period in periods:
            #filepath = f"modules/SENTINEL2/prediction_final_{year}_{period}.tif"
            filepath = os.path.join(BASE_MODULES, 'SENTINEL2', f"prediction_final_{year}_{period}.tif")
            try:
                # --- ekstraksi ---
                ocean_mask, contours_pixel, contours_geo, meta, array, fig = coastline.extract_coastline_from_geotiff(
                    filepath,
                    year,
                    period,
                    water_value=1,
                    land_value=0,
                    ws = 7
                )
                transform = meta["transform"]
                new_period = convert_q_to_text(period)

                coastlines_all.append({
                    "year": year,
                    "period": new_period,
                    "group_name": f"{year}_{new_period}",
                    "mask": array,
                    "transform": transform,
                    "coastline": contours_geo,
                    "plot": fig
                })
                plt.close(fig)
                numPlot+=1
            except Exception as e:
                continue
                # print(f"Gagal baca {filepath}: {e}")
        listPlot.append(numPlot)
        # print(listPlot) 

    return coastlines_all, listPlot

def generate_coastline_all():
    coastlines_all, listPlot = init_result()
    
    # --- Setup warna ---
    years = sorted(set([c["year"] for c in coastlines_all]))

    # Buat gradasi warna untuk seluruh tahun dalam range 0.3–1.0
    cmap = cm.Blues
    colors = cmap(np.linspace(0.3, 1.0, len(years)))
    # Mapping: tahun -> warna
    color_map = {year: colors[i] for i, year in enumerate(years)}
    
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

def generate_coastline_compare(startYear, endYear, coastlines_all):
    # Tahun yang dipakai
    years = [startYear, endYear + 1]
    periods = ["Jan_Jun", "Jul_Des", "Jan_Mar", "Apr_Jun", "Jul_Sep", "Okt_Des"]

    # Total kombinasi
    total_items = len(years) * len(periods)
    cmap = cm.Blues  # pilih palette
    grad_colors = cmap(np.linspace(0.3, 1.0, total_items))  # 30%–100% supaya kontras

    # === Mapping kombinasi (year, period) -> warna ===
    color_map = {}
    idx = 0
    for y in years:
        for p in periods:
            color_map[(y, p)] = grad_colors[idx]
            idx += 1

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

    # out_path = os.path.join(OUTPUT_DIR, f"coastline_compare_{startYear}-{endYear}.png")
    plt.savefig(f'../web_app/static/assets/compare/predictionAll.png', dpi=300, bbox_inches='tight')
    # plt.show()

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
    
    # Buat gradasi warna untuk seluruh tahun dalam range 0.3–1.0
    cmap = cm.Blues
    colors = cmap(np.linspace(0.3, 1.0, len(avg_coastlines)))

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
    # plt.show()

## Harusnya ini udh gadipake lagi (pake dari axel)
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
    # out_path = os.path.join(OUTPUT_DIR, f"coastline_avg_{curYear}-{curYear+1}.png")
    # plt.savefig(out_path, dpi=300, bbox_inches='tight')
    # plt.show()

def fig_to_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

def generate_prediction_all_by_year(year):
    coastlines_all, listPlot = init_result()
    # filter tahun
    data_year = [c for c in coastlines_all if c["year"] == year]
    n_fig = listPlot[year-2013]

    # layout subplot: max 4 (2x2)
    rows = 2 if n_fig > 2 else 1
    cols = 2 if n_fig > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    axes = np.array(axes).flatten()

    # sort periods agar berurutan
    period_order = ["Jan_Jun", "Jul_Des", "Jan_Mar", "Apr_Jun", "Jul_Sep", "Okt_Des"]
    data_year = sorted(data_year, key=lambda x: period_order.index(x["period"]))

    for i in range(n_fig):
        fig_old = data_year[i]["plot"]
        period = data_year[i]["period"]

        img = fig_to_array(fig_old)  # convert fig to array

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(period)

    # kosongkan subplot yang tidak terpakai
    for j in range(n_fig, len(axes)):
        axes[j].axis("off")

    # judul besar
    # plt.suptitle(f"prediction_all - {year}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # simpan file
    out_file = os.path.join(OUTPUT_DIR_2, f"prediction_all_{year}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_coastlines_all_by_year(year):
    coastlines_all = generate_coastline_all()

    # Ambil data untuk tahun tertentu
    data_year = [c for c in coastlines_all if c["year"] == year]
    if not data_year:
        print(f"[WARN] Tidak ada data coastline untuk tahun {year}")
        return None

    # Perioda yang kemungkinan muncul (urutkan supaya konsisten)
    periods = ["Jan_Jun", "Jul_Des", "Jan_Mar", "Apr_Jun", "Jul_Sep", "Okt_Des"]

    # Warna: satu warna per perioda (gradasi)
    cmap = cm.Blues
    # grad_colors = cmap(np.linspace(0.3, 1.0, len(periods)))
    grad_colors = cmap([0.75, 1, 0.475, 0.65, 0.825, 1])
    color_map = {periods[i]: grad_colors[i] for i in range(len(periods))}

    # Buat figure
    plt.figure(figsize=(12, 10))

    for item in data_year:
        period = item["period"]
        coastline = item["coastline"]

        # jika period tidak di daftar periods, tambahkan ke mapping warna dengan next color
        if period not in color_map:
            # fallback: gunakan last color
            color = grad_colors[-1]
        else:
            color = color_map[period]

        for contour in coastline:
            xs = [pt[0] for pt in contour]
            ys = [pt[1] for pt in contour]
            plt.plot(xs, ys, color=color, linewidth=1, label=f"{period} {year}")

    # Hilangkan duplikat legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(f"Coastlines {year} (semua perioda)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")

    out_path = os.path.join(OUTPUT_DIR, f"coastline_year_{year}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    return out_path    

def measure(lat1, lon1, lat2, lon2):
    """Menghitung jarak dua koordinat (lat, lon) dalam meter menggunakan rumus Haversine."""
    R = 6378.137  # radius Bumi dalam KM

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)

    a = (
        math.sin(dLat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dLon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = R * c  # dalam KM
    return d * 1000  # ubah jadi meter

def find_index_pair(first_line, last_line, num_samples=5):
    """
    Mencari pasangan index dari dua garis pantai berdasarkan kesamaan longitude.

    Parameters:
    -----------
    first_line : np.array
        Array koordinat garis pantai pertama (longitude, latitude)
    last_line : np.array
        Array koordinat garis pantai terakhir (longitude, latitude)
    num_samples : int
        Jumlah sampel yang ingin diambil

    Returns:
    --------
    matched_indices : list of tuples
        List berisi tuple (idx_first, idx_last) yang merupakan pasangan index
    """
    # Sampling index dari garis pertama
    sample_indices_first = np.linspace(0, len(first_line)-1, num_samples).astype(int)
    matched_indices = []
    for idx_first in sample_indices_first:
        lon_first = first_line[idx_first, 0]

        # Cari index pada garis terakhir yang memiliki longitude terdekat
        lon_differences = np.abs(last_line[:, 0] - lon_first)
        idx_last = np.argmin(lon_differences)

        matched_indices.append((idx_first, idx_last))

    return matched_indices

def plot_coastline_distances(method, data, num_samples=8):
    """
    Memplot jarak antara dua garis pantai dengan matching longitude yang lebih akurat.

    Parameters:
    -----------
    data : list of dictionaries
        List dengan key "mean_coastline"
    num_samples : int
        Berapa banyak garis jarak yang ingin ditampilkan
    """
    # Ambil garis pertama dan terakhir
    first = np.array(data[0]["coastline"])
    last  = np.array(data[-1]["coastline"])

    # Cari pasangan index dengan longitude terdekat
    matched_indices = find_index_pair(first, last, num_samples)
    # Hitung jarak untuk setiap pasangan titik
    distances = []
    for idx_first, idx_last in matched_indices:
        dist = measure(first[idx_first, 1], first[idx_first, 0],
                      last[idx_last, 1], last[idx_last, 0])
        distances.append(dist)

    # ----- PLOTTING -----
    plt.figure(figsize=(10, 8))

    # Definisi color map
    cmap = plt.get_cmap('Blues')
    coastline_colors = cmap(np.linspace(0.3, 1.0, len(data)))

    for i in range (len(data)):
      coastline = data[i]['coastline']
      plt.plot(coastline[:, 0], coastline[:, 1],
               linewidth=2,
               color=coastline_colors[i],
               label=f"Garis Pantai {data[i]['group_name']}")

    for idx, (idx_first, idx_last) in enumerate(matched_indices):
        x_vals = [first[idx_first, 0], last[idx_last, 0]]
        y_vals = [first[idx_first, 1], last[idx_last, 1]]

        plt.plot(x_vals, y_vals, '--', linewidth=1.5, color='black', alpha=0.7)

        # Tambahkan marker pada titik-titik
        plt.plot(first[idx_first, 0], first[idx_first, 1], 'bo', markersize=6)
        plt.plot(last[idx_last, 0], last[idx_last, 1], 'ro', markersize=6)

        # Tambahkan background putih untuk teks agar lebih mudah dibaca
        plt.text(x_vals[1], y_vals[0], f"{distances[idx]:.2f} m",
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.title("Jarak Antar Garis Pantai", fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)

    if(method == "all"):
        plt.savefig(f'../web_app/static/assets/compare/predictionAll.png', dpi=300, bbox_inches='tight')
    elif(method == "avg"):
        plt.savefig(f'../web_app/static/assets/compare/predictionAvg.png', dpi=300, bbox_inches='tight')
    # plt.show()

def generate_coastline_compare_new(startYear, endYear, coastlines_all):
    chosenYear = range(startYear, endYear + 1)
    filtered_coastlines = []
    for year in chosenYear:
        for c in coastlines_all:
            if(c["year"] == year):
                filtered_coastlines.append({
                    "year": c["year"],
                    "period": c["period"],
                    "group_name": c["group_name"],
                    "mask": c["mask"],
                    "transform": c["transform"],
                    "coastline": interpolate_line(c["coastline"][0], num_points=1000),
                    "plot": c["plot"]
                })
    plot_coastline_distances("all", filtered_coastlines, num_samples=8)

def generate_coastline_compare_average(startYear, endYear, coastlines_all, num_points=1000):
    chosenYear = range(startYear, endYear + 1)
    year_group = {}
    avg_coastlines = []

    # Kelompokkan tahun
    year_start = startYear
    year_end = endYear + 1
    increment = 1

    while (year_start < year_end):
        now = int(year_start + increment - 1)
        year_group[f'{year_start}'] = [y for y in chosenYear if y >= year_start and y <= now]
        year_start = int(now + 1)

    # Hitung rata-rata garis pantai per kelompok tahun
    for group_name, group_years in year_group.items():
        coastlines = []
        for c in coastlines_all:
            if(c["year"] in group_years):
                coastlines.append(interpolate_line(c["coastline"][0], num_points))
        mean_coastline = np.mean(coastlines, axis=0)
        avg_coastlines.append({
            "group_name": group_name,
            "coastline": mean_coastline
        })
    
    # Plot jarak antar garis pantai rata-rata
    plot_coastline_distances("avg", avg_coastlines, num_samples=8)