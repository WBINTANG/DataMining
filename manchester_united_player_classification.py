# Import library yang dibutuhkan
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Pengumpulan Data
data = {
    'nama': ['Marcus Rashford', 'Bruno Fernandes', 'Harry Maguire', 'David De Gea', 'Jadon Sancho', 'Casemiro'],
    'tinggi_cm': [185, 179, 194, 192, 180, 185],
    'berat_kg': [70, 69, 100, 83, 76, 84],
    'gol': [17, 9, 1, 0, 6, 5],
    'assist': [9, 12, 1, 0, 4, 2],
    'penyelamatan': [0, 0, 0, 95, 0, 0],
    'posisi': ['FW', 'MF', 'DF', 'GK', 'FW', 'MF']
}

df = pd.DataFrame(data)

# 2. Pembersihan Data
print(df.isnull().sum())  # Cek apakah ada data kosong

# 3. Visualisasi Data
sns.pairplot(df, hue='posisi')

# 4. Pemodelan (Random Forest)
X = df[['tinggi_cm', 'berat_kg', 'gol', 'assist', 'penyelamatan']]
y = df['posisi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = model.predict(X_test)
print(f'Akurasi model: {accuracy_score(y_test, y_pred) * 100:.2f}%')
