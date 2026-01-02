# proyek-ds-2
1. 6182201017 / Tiffany Tasya
2. 6182201053 / Axel Darmaputra
3. 6182201054 / Junita Hariyati
4. 6182201085 / Johnson Natanael 

## Spesifikasi Sistem
- **Python** - bahasa pemrograman yang digunakan
- **Google Earth Engine** - aplikasi untuk menjalakan model

## Penggunaan  
Clone proyek proyek-ds-2
```
git clone https://github.com/triyudhaa/proyek-ds-2.git 
```
Pindah ke dalam direktori proyek proyek-ds-2
```
cd proyek-ds-2
```

# Instalasi dan Pengaturan
## 1. Persiapan Database 
1. Pengguna harus memiliki akun GEE dengan mendaftarkan menggunakan akun Google atau membuat akun baru pada link `https://code.earthengine.google.com`
2. Instalasi library Python yang akan dibutuhkan. Pastikan perangkat keras yang digunakan sudah memiliki instalasi bahasa Python. 
```
pip install rasterio 
pip install scipy 
pip install scikit-image
```
3. Pengguna mengganti parameter project pada file `sentinel_model.py` dan `landsat_model.py` sesuai dengan **nama akun GEE** yang terdaftar. Nama akun GEE dapat dilihat pada pojok kanan halaman `https://code.earthengine.google.com`
4. Pengguna menjalankan perangkat lunak dengan pindah direktori pada folder web_app dan perintah running Python. 
```
cd web_app
python app.py
```
5. Perangkat lunak akan berjalan dengan mengakses `http://127.0.0.1:5000` pada browser.