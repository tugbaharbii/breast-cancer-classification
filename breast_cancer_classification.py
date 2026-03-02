# =============================================================================
# Meme Kanseri Sınıflandırması - Random Forest
# Breast Cancer Wisconsin Dataset
#
# Öğrenci No : _______________
# Ad Soyad   : _______________
# GitHub     : github.com/_______________
# =============================================================================

# ── 1. Kütüphane İmportları ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── 2. Veri Yükleme ──────────────────────────────────────────────────────────
# Dosya yolunu kendi sisteminize göre düzenleyin
yol = r'C:\Users\hp\Downloads\data.xlsx'
df = pd.read_excel(yol)


print("=" * 60)
print("VERİ SETİ GENEL BİLGİLER")
print("=" * 60)
print(f"Satır sayısı  : {df.shape[0]}")
print(f"Sütun sayısı  : {df.shape[1]}")
print(f"\nSınıf Dağılımı:\n{df['diagnosis'].value_counts()}")
print(f"\nİlk 5 Satır:\n{df.head()}")

# ── 3. Ön İşleme ─────────────────────────────────────────────────────────────
# Gereksiz sütunu kaldır
df = df.drop(columns=["id"])

# Hedef değişkeni sayısala çevir: M (Malignant/Kötü Huylu) = 1, B (Benign/İyi Huylu) = 0
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Tüm öznitelik sütunlarını sayısala zorla, hatalı değerleri NaN yap
for col in df.columns:
    if col != "diagnosis":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Eksik değer içeren satırları temizle
df = df.dropna()

print(f"\nTemizlik sonrası satır sayısı: {df.shape[0]}")

# ── 4. Öznitelik / Hedef Ayrımı ──────────────────────────────────────────────
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# ── 5. Eğitim / Test Ayrımı (%80 - %20) ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEğitim seti boyutu : {X_train.shape[0]} örnek")
print(f"Test seti boyutu   : {X_test.shape[0]} örnek")

# ── 6. Model Eğitimi: Random Forest ──────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel başarıyla eğitildi!")

# ── 7. Tahmin ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── 8. Değerlendirme Metrikleri ───────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("DEĞERLENDİRME METRİKLERİ")
print("=" * 60)
print(f"Accuracy  (Doğruluk)    : {accuracy:.4f}  (%{accuracy*100:.2f})")
print(f"Precision (Kesinlik)    : {precision:.4f}  (%{precision*100:.2f})")
print(f"Recall    (Duyarlılık)  : {recall:.4f}  (%{recall*100:.2f})")
print(f"F1-Score                : {f1:.4f}  (%{f1*100:.2f})")
print("\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Benign (İyi Huylu)", "Malignant (Kötü Huylu)"]))

# ── 9. GÖRSELLEŞTİRME ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Meme Kanseri Sınıflandırması - Random Forest Sonuçları", fontsize=15, fontweight="bold")

# ── 9a. Sınıf Dağılımı (Pasta Grafik) ────────────────────────────────────────
class_counts = y.value_counts()
axes[0].pie(
    class_counts,
    labels=["Benign (İyi Huylu)", "Malignant (Kötü Huylu)"],
    autopct="%1.1f%%",
    colors=["#4CAF50", "#F44336"],
    startangle=90,
    explode=(0.05, 0.05),
)
axes[0].set_title("Sınıf Dağılımı", fontsize=13)

# ── 9b. Confusion Matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1],
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    linewidths=0.5,
)
axes[1].set_title("Confusion Matrix", fontsize=13)
axes[1].set_xlabel("Tahmin Edilen Sınıf")
axes[1].set_ylabel("Gerçek Sınıf")

# ── 9c. Feature Importance (Top 15) ──────────────────────────────────────────
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top15 = feature_importances.sort_values(ascending=True).tail(15)

colors_bar = ["#1565C0" if i >= 12 else "#42A5F5" for i in range(15)]
top15.plot(kind="barh", ax=axes[2], color=colors_bar)
axes[2].set_title("En Önemli 15 Öznitelik", fontsize=13)
axes[2].set_xlabel("Önem Skoru")
axes[2].set_ylabel("Öznitelik")

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGörseller 'results.png' olarak kaydedildi.")

# ── 10. Metrik Özet Grafik ────────────────────────────────────────────────────
metrics = {
    "Accuracy\n(Doğruluk)": accuracy,
    "Precision\n(Kesinlik)": precision,
    "Recall\n(Duyarlılık)": recall,
    "F1-Score": f1,
}

fig2, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics.keys(), metrics.values(), color=["#1565C0", "#1976D2", "#1E88E5", "#42A5F5"], width=0.5)
ax.set_ylim(0, 1.15)
ax.set_title("Model Performans Metrikleri - Random Forest", fontsize=13, fontweight="bold")
ax.set_ylabel("Skor")

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="%95 Referans Çizgisi")
ax.legend()
plt.tight_layout()
plt.savefig("metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Metrik grafiği 'metrics.png' olarak kaydedildi.")

print("\n" + "=" * 60)
print("İŞLEM TAMAMLANDI")
print("=" * 60)