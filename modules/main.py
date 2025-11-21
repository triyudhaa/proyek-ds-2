import sentinel_model
import coastline
import sentinel_hasil
import landsat_model
import landsat_hasil
import combine_hasil

# General Coastline Landsat 
# combine_hasil.generate_coastline_all()
combine_hasil.avg_coastline(2)
# combine_hasil.avg_coastline(4)
# combine_hasil.avg_coastline(6)
# combine_hasil.generate_coastline_compare(2013)
# combine_hasil.generate_coastline_compare(2023)
# combine_hasil.generate_coastline_compare_avg(2013)
# combine_hasil.generate_coastline_compare_avg(2023)

# General Coastline Sentinel-2
# coastline_sentinel = sentinel_hasil.generate_coastline_sentinel()

startDate = '2024-01-01'
endDate = '2024-07-01'

# Hasil Predict Model Sentinel
# output: plot show prediction raw
# filename = sentinel_model.init_predict_sentinel(startDate, endDate)

# Hasil Coastline dari Prediksi
# output: plot show prediction smoothing & plot coastline
# coastline.extract_coastline_from_input(
#     filename,
#     startDate,
#     endDate,
#     water_value=1,
#     land_value=0,
#     ws = 7
# )

# General Coastline Landsat 
# coastline_landsat = landsat_hasil.generate_coastline_landsat()

startDate = '2013-01-01'
endDate = '2013-07-01'

# Hasil Predict Model Landsat
# output: plot show prediction raw
# filename = landsat_model.init_predict_landsat(startDate, endDate)

# Hasil Coastline dari Prediksi
# output: plot show prediction smoothing & plot coastline
# coastline.extract_coastline_from_input(
#     filename,
#     startDate,
#     endDate,
#     water_value=1,
#     land_value=0,
#     ws = 7
# )