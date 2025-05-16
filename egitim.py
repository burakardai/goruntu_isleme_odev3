from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report # Daha detaylı rapor için
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# Veri setini yükle
try:
    df = pd.read_csv("veriseti.csv")
except FileNotFoundError:
    print("Hata: 'veriseti.csv' dosyası bulunamadı. Lütfen önce yuz_algila.py ile veri toplayın.")
    exit()

if df.empty:
    print("Hata: 'veriseti.csv' dosyası boş. Lütfen veri toplayın.")
    exit()

# Özellikler (X) ve Etiket (y) ayırma
try:
    y = df["Etiket"]
    X = df.drop("Etiket", axis=1)
except KeyError:
    print("Hata: 'veriseti.csv' dosyasında 'Etiket' sütünü bulunamadı veya format hatalı.")
    exit()

# Sütun isimlerinin string olduğundan emin ol
X.columns = X.columns.astype(str)

# Eğitim ve test olarak 2'ye böl
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Toplam örnek sayısı: {len(df)}")
print(f"Eğitim seti örnek sayısı: {len(X_egitim)}")
print(f"Test seti örnek sayısı: {len(X_test)}")
print(f"Eğitimdeki etiket dağılımı:\n{y_egitim.value_counts(normalize=True)}")
print(f"Testteki etiket dağılımı:\n{y_test.value_counts(normalize=True)}")


# Pipeline oluşturma
pipeline = Pipeline([
    ("std", StandardScaler()),  # Veriyi ölçeklendir
    ("sınıflandırıcı", LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000))
])

# Modeli eğitim verisetini kullanarak eğit
print("\nModel eğitiliyor...")
pipeline.fit(X_egitim, y_egitim)
print("Model eğitimi tamamlandı.")

# Modelin test setindeki tahminlerini al
y_tahmin = pipeline.predict(X_test)

# Doğruluk oranını hesapla
dogruluk_orani = accuracy_score(y_test, y_tahmin)
print(f"\nModel Doğruluk Oranı = {dogruluk_orani:.4f}")

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin))

# Eğitilmiş modeli disk üzerinde "model.pkl" adıyla kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("\nEğitilmiş model 'model.pkl' olarak kaydedildi.")