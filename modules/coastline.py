import rasterio 
import numpy as np
from io import BytesIO
from scipy import ndimage
from skimage import measure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
random.seed(42)
np.random.seed(42)

"""Melakukan proses sliding window untuk smoothing hasil prediksi"""
def sliding_window_majority(arr, window_size=7):
    pad = window_size // 2
    rows, cols = arr.shape
    result = np.zeros_like(arr)

    for i in range(rows):
        for j in range(cols):
            r_start, r_end = max(0, i-pad), min(rows, i+pad+1)
            c_start, c_end = max(0, j-pad), min(cols, j+pad+1)

            window = arr[r_start:r_end, c_start:c_end]

            # penentuan mayoritas
            ones = np.sum(window)
            zeros = window.size - ones
            result[i, j] = 1 if ones >= zeros else 0

    return result

"""Membersihkan noise kecil pada hasil prediksi dengan flood fill"""
def clean_mask(mask, target_value, min_size):
  # ambil kelas yang dipilih
  binary = (mask == target_value).astype(np.uint8)

  labeled, num_features = ndimage.label(binary)
  sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]

  for region_id, size in enumerate(sizes, start=1):
    if size < min_size:
        # hapus komponen kecil
        mask[labeled == region_id] = 1 - target_value

  return mask

"""Membaca file GeoTIFF"""
def read_geotiff(filepath):
    with rasterio.open(filepath) as src:
        # Baca band pertama
        array = src.read(1)

        # Simpan metadata
        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'nodata': src.nodata
        }

    return array, meta

"""Ekstraksi garis pantai dari file GeoTIFF untuk Sentinel"""
def extract_coastline_from_geotiff(filepath, year, period, water_value=1, land_value=0, ws = 7):
    array, meta = read_geotiff(filepath)

    if meta['nodata'] is not None:
        array = np.where(array == meta['nodata'], land_value, array)

    # koreksi sliding window dan flood fill
    array = clean_mask(array, target_value=1, min_size=10000)
    array = clean_mask(array, target_value=0, min_size=500)
    array = sliding_window_majority(array, window_size=ws)

    water_mask = (array == water_value)
    
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(array, cmap="coolwarm")
    # ax.axis("off")
    # ax.set_title(f"Prediction After Smoothing - {year}_{period}")
    
    # tampilkan visualisasi hasil koreksi
    cmap = colors.ListedColormap(["#B40B27", "#3C4DC1"])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(array, cmap=cmap, norm=norm)
    ax.axis("off")
    ax.set_title(f"Prediction After Smoothing - {year}_{period}")

    # pelabelan badan air selain laut dengan nilai lain
    labeled_water, num_features = ndimage.label(water_mask)

    # identifikasi label yang menyentuh tepi
    ocean_labels = set()
    h, w = array.shape

    # cek tepian
    ocean_labels.update(labeled_water[0, :])  # tepi atas
    ocean_labels.update(labeled_water[-1, :])  # tepi bawah
    ocean_labels.update(labeled_water[:, 0])  # tepi kiri
    ocean_labels.update(labeled_water[:, -1])  # tepi kanan

    # hapus background
    ocean_labels.discard(0)

    # simpan hasil mask laut
    ocean_mask = np.isin(labeled_water, list(ocean_labels))

    # deteksi kontur untuk membuat garis pantai
    contours = measure.find_contours(ocean_mask.astype(float), 0.5)

    # ubah koordinat piksel ke koordinat geografis (longitude, latitude)
    contours_geo = []
    transform = meta['transform']

    for contour in contours:
        geo_coords = []
        for row, col in contour:
            # Konversi pixel ke koordinat geografis
            lon, lat = rasterio.transform.xy(transform, row, col)
            geo_coords.append((lon, lat))
        contours_geo.append(np.array(geo_coords))

    return ocean_mask, contours, contours_geo, meta, array, fig

"""Ekstraksi garis pantai dari file GeoTIFF untuk Landsat"""
def extract_coastline_from_geotiff_landsat(filepath, year, period, water_value=1, land_value=0, ws = 7):
    raw, meta = read_geotiff(filepath)

    if meta['nodata'] is not None:
        raw = np.where(raw == meta['nodata'], land_value, raw)

    # koreksi sliding window dan flood fill
    array = clean_mask(raw, target_value=1, min_size=5000)
    array = clean_mask(array, target_value=0, min_size=500)
    array = sliding_window_majority(array, window_size=ws)
    
    # plot sebelum  smoothing 
    # plt.figure(figsize=(8, 8))
    # plt.imshow(raw, cmap="coolwarm")
    # plt.axis('off')
    # plt.title(f"Prediction Before Smoothing - {year}_{period}")
    # plt.show()
    
    # plot setelah smoothing 
    # plt.figure(figsize=(8, 8))
    # plt.imshow(array, cmap="coolwarm")
    # plt.axis('off')
    # plt.title(f"Prediction After Smoothing - {year}_{period}")
    # plt.show()

    # deteksi kontur
    contours = measure.find_contours(array.astype(float), 0.5)
    transform = meta['transform']

    # ubah koordinat piksel ke koordinat geografis (longitude, latitude)
    contours_geo = []
    for contour in contours:
        geo_coords = []
        for row, col in contour:
            lon, lat = rasterio.transform.xy(transform, row, col)
            geo_coords.append((lon, lat))
        contours_geo.append(np.array(geo_coords))

    return contours, contours_geo, meta, array

"""Ekstraksi garis pantai dari input custom user"""
def extract_coastline_from_input(filepath, startDate, endDate, water_value=1, land_value=0, ws = 7):
    array, meta = read_geotiff(filepath)
    array = np.where(array == 58, 1, 0).astype(np.uint8)

    if meta['nodata'] is not None:
        array = np.where(array == meta['nodata'], land_value, array)

    # koreksi sliding window dan flood fill
    array = clean_mask(array, target_value=1, min_size=7000)
    array = clean_mask(array, target_value=0, min_size=500)
    array = sliding_window_majority(array, window_size=ws)

    water_mask = (array == water_value)
    
    # plot setelah smoothing 
    plt.figure(figsize=(8, 8))
    plt.imshow(array, cmap="coolwarm")
    plt.axis('off')
    plt.title(f"Prediksi {startDate} sampai {endDate}")
    plt.savefig(f'../web_app/static/assets/custom_model/prediction.png',
                dpi=300, 
                bbox_inches='tight')
    # plt.show()

    # pelabelan badan air selain laut dengan nilai lain
    labeled_water, num_features = ndimage.label(water_mask)

    # identifikasi label yang menyentuh tepi
    ocean_labels = set()
    h, w = array.shape

    # cek tepian
    ocean_labels.update(labeled_water[0, :])  # tepi atas
    ocean_labels.update(labeled_water[-1, :])  # tepi bawah
    ocean_labels.update(labeled_water[:, 0])  # tepi kiri
    ocean_labels.update(labeled_water[:, -1])  # tepi kanan

    # hapus background
    ocean_labels.discard(0)

    # simpan hasil mask laut
    ocean_mask = np.isin(labeled_water, list(ocean_labels))

    # deteksi kontur untuk membuat garis pantai
    contours = measure.find_contours(ocean_mask.astype(float), 0.5)

    # ubah koordinat piksel ke koordinat geografis (longitude, latitude)
    contours_geo = []
    transform = meta['transform']

    for contour in contours:
        geo_coords = []
        for row, col in contour:
            lon, lat = rasterio.transform.xy(transform, row, col)
            geo_coords.append((lon, lat))
        contours_geo.append(np.array(geo_coords))

    plt.figure(figsize=(10, 8))

    for contour in contours_geo:
        xs = [pt[0] for pt in contour]
        ys = [pt[1] for pt in contour]  
        plt.plot(xs, ys, linewidth=2, label=f"{startDate} {endDate}")

    plt.title(f"Garis Pantai {startDate} sampai {endDate}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # plt.show()
    plt.savefig(f'../web_app/static/assets/custom_model/coastline.png',
            dpi=300, 
            bbox_inches='tight')