import pandas as pd

# Membaca data dari file CSV
df = pd.read_csv('makanan.csv')

# Melakukan preprocessing atau manipulasi data jika diperlukan
# ...

# Contoh manipulasi data: mengambil kolom "Name"
df = df[['Name']]