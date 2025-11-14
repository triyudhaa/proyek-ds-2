import numpy as np
import matplotlib.pyplot as plt
import random
import rasterio 
from skimage import measure
random.seed(42)
np.random.seed(42)

def generate_coastline_2lines(coastlines_all):
    avg_coastlines_all = []
    
    # Ambil semua tahun unik dari corrections_all
    years = sorted(set([c["year"] for c in coastlines_all]))

    # === Bagi ke dalam dua kelompok 6 tahun ===
    year_groups = {
        "2013–2018": [y for y in years if 2013 <= y <= 2018],
        "2019–2024": [y for y in years if 2019 <= y <= 2024]
    }

    for group_name, group_years in year_groups.items():
        # Ambil semua mask untuk tahun-tahun dalam kelompok ini
        masks = [c["mask"] for c in coastlines_all if c["year"] in group_years]
        transforms = [c["transform"] for c in coastlines_all if c["year"] in group_years]

        if len(masks) == 0:
            print(f"⚠️ Tidak ada data untuk {group_name}, dilewati.")
            continue

        # === 1. Hitung rata-rata antar semua mask dalam kelompok ===
        avg_mask = np.mean(masks, axis=0)

        # === 2. Threshold 0.5 (>=0.5 dianggap air) ===
        binary_avg = (avg_mask >= 0.5).astype(np.uint8)

        # === 3. Ambil contour (garis pantai rata-rata kelompok) ===
        contours = measure.find_contours(binary_avg, level=0.5)

        # Ambil transform dari salah satu mask (anggap sama)
        transform = transforms[0]

        # Konversi pixel -> koordinat (longitude, latitude)
        coastlines = []
        for contour in contours:
            xs = contour[:, 1]
            ys = contour[:, 0]
            longs, lats = rasterio.transform.xy(transform, ys, xs)
            coastlines.append(list(zip(longs, lats)))

        avg_coastlines_all.append({
            "period": group_name,
            "coastline": coastlines
        })

    # === Plot hasil coastline rata-rata per 6 tahun ===
    colors = plt.cm.plasma(np.linspace(0, 1, len(avg_coastlines_all)))
    plt.figure(figsize=(12, 10))

    for i, item in enumerate(avg_coastlines_all):
        period = item["period"]
        coastline = item["coastline"]

        for contour in coastline:
            xs = [pt[0] for pt in contour]
            ys = [pt[1] for pt in contour]
            plt.plot(xs, ys, color=colors[i], linewidth=1.8, label=period)

    # Hilangkan label duplikat di legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("Rata-Rata Garis Pantai per 6 Tahun")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.savefig("coastline_2lines.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def generate_coastline_4lines(coastlines_all):
    avg_coastlines_all = []
    
    # Ambil semua tahun unik dari corrections_all
    years = sorted(set([c["year"] for c in coastlines_all]))

    # === Bagi ke dalam dua kelompok 6 tahun ===
    year_groups = {
        "2013–2015": [y for y in years if 2013 <= y <= 2015],
        "2016–2018": [y for y in years if 2016 <= y <= 2018],
        "2019–2021": [y for y in years if 2019 <= y <= 2021],
        "2021–2024": [y for y in years if 2021 <= y <= 2024]
    }

    for group_name, group_years in year_groups.items():
        # Ambil semua mask untuk tahun-tahun dalam kelompok ini
        masks = [c["mask"] for c in coastlines_all if c["year"] in group_years]
        transforms = [c["transform"] for c in coastlines_all if c["year"] in group_years]

        if len(masks) == 0:
            print(f"⚠️ Tidak ada data untuk {group_name}, dilewati.")
            continue

        # === 1. Hitung rata-rata antar semua mask dalam kelompok ===
        avg_mask = np.mean(masks, axis=0)

        # === 2. Threshold 0.5 (>=0.5 dianggap air) ===
        binary_avg = (avg_mask >= 0.5).astype(np.uint8)

        # === 3. Ambil contour (garis pantai rata-rata kelompok) ===
        contours = measure.find_contours(binary_avg, level=0.5)

        # Ambil transform dari salah satu mask (anggap sama)
        transform = transforms[0]

        # Konversi pixel -> koordinat (longitude, latitude)
        coastlines = []
        for contour in contours:
            xs = contour[:, 1]
            ys = contour[:, 0]
            longs, lats = rasterio.transform.xy(transform, ys, xs)
            coastlines.append(list(zip(longs, lats)))

        avg_coastlines_all.append({
            "period": group_name,
            "coastline": coastlines
        })

    # === Plot hasil coastline rata-rata per 6 tahun ===
    colors = plt.cm.plasma(np.linspace(0, 1, len(avg_coastlines_all)))
    plt.figure(figsize=(12, 10))

    for i, item in enumerate(avg_coastlines_all):
        period = item["period"]
        coastline = item["coastline"]

        for contour in coastline:
            xs = [pt[0] for pt in contour]
            ys = [pt[1] for pt in contour]
            plt.plot(xs, ys, color=colors[i], linewidth=1.8, label=period)

    # Hilangkan label duplikat di legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("Rata-Rata Garis Pantai per 3 Tahun")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.savefig("coastline_4lines.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def generate_coastline_6lines(coastlines_all):
    avg_coastlines_all = []
    
    # Ambil semua tahun unik dari corrections_all
    years = sorted(set([c["year"] for c in coastlines_all]))

    # === Bagi ke dalam dua kelompok 6 tahun ===
    year_groups = {
        "2013–2014": [y for y in years if 2013 <= y <= 2014],
        "2015–2016": [y for y in years if 2015 <= y <= 2016],
        "2017–2018": [y for y in years if 2017 <= y <= 2018],
        "2019–2020": [y for y in years if 2019 <= y <= 2020],
        "2021–2022": [y for y in years if 2021 <= y <= 2022],
        "2023–2024": [y for y in years if 2023 <= y <= 2024]
    }

    for group_name, group_years in year_groups.items():
        # Ambil semua mask untuk tahun-tahun dalam kelompok ini
        masks = [c["mask"] for c in coastlines_all if c["year"] in group_years]
        transforms = [c["transform"] for c in coastlines_all if c["year"] in group_years]

        if len(masks) == 0:
            print(f"⚠️ Tidak ada data untuk {group_name}, dilewati.")
            continue

        # === 1. Hitung rata-rata antar semua mask dalam kelompok ===
        avg_mask = np.mean(masks, axis=0)

        # === 2. Threshold 0.5 (>=0.5 dianggap air) ===
        binary_avg = (avg_mask >= 0.5).astype(np.uint8)

        # === 3. Ambil contour (garis pantai rata-rata kelompok) ===
        contours = measure.find_contours(binary_avg, level=0.5)

        # Ambil transform dari salah satu mask (anggap sama)
        transform = transforms[0]

        # Konversi pixel -> koordinat (longitude, latitude)
        coastlines = []
        for contour in contours:
            xs = contour[:, 1]
            ys = contour[:, 0]
            longs, lats = rasterio.transform.xy(transform, ys, xs)
            coastlines.append(list(zip(longs, lats)))

        avg_coastlines_all.append({
            "period": group_name,
            "coastline": coastlines
        })

    # === Plot hasil coastline rata-rata per 6 tahun ===
    colors = plt.cm.plasma(np.linspace(0, 1, len(avg_coastlines_all)))
    plt.figure(figsize=(12, 10))

    for i, item in enumerate(avg_coastlines_all):
        period = item["period"]
        coastline = item["coastline"]

        for contour in coastline:
            xs = [pt[0] for pt in contour]
            ys = [pt[1] for pt in contour]
            plt.plot(xs, ys, color=colors[i], linewidth=1.8, label=period)

    # Hilangkan label duplikat di legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("Rata-Rata Garis Pantai per 2 Tahun")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.savefig("coastline_6lines.png", dpi=300, bbox_inches='tight')
    plt.show()