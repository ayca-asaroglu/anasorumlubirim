import pandas as pd
import numpy as np

# Test verisi oluştur
np.random.seed(42)

# Sınıf isimleri
upper_classes = ['IT', 'HR', 'Finance', 'Operations', 'Marketing']
lower_classes = {
    'IT': ['IT_Support', 'IT_Development', 'IT_Infrastructure'],
    'HR': ['HR_Recruitment', 'HR_Payroll', 'HR_Training'],
    'Finance': ['Finance_Accounting', 'Finance_Budgeting'],
    'Operations': ['Operations_Logistics', 'Operations_Quality'],
    'Marketing': ['Marketing_Digital', 'Marketing_Brand']
}

# Test metinleri
test_texts = [
    "Sistemde bir hata oluştu ve kullanıcılar giriş yapamıyor. Lütfen bu sorunu çözün.",
    "Yeni kullanıcı kaydı gerekiyor. Form doldurulması ve onay süreci başlatılmalı.",
    "Muhasebe raporu hazırlanması gerekiyor. Aylık finansal durum analizi yapılacak.",
    "Web sitesi güncellemesi yapılması gerekiyor. Yeni özellikler eklenmeli.",
    "Personel eğitimi düzenlenmesi gerekiyor. Yeni çalışanlar için oryantasyon programı.",
    "Sistem yedekleme işlemi başarısız oldu. Veri kaybı riski var.",
    "Bütçe planlaması yapılması gerekiyor. Yeni yıl için mali plan hazırlanacak.",
    "Lojistik operasyonları optimize edilmesi gerekiyor. Teslimat süreleri uzun.",
    "Dijital pazarlama kampanyası başlatılması gerekiyor. Sosyal medya stratejisi.",
    "Veritabanı performans sorunu var. Sorgu süreleri çok uzun.",
    "İnsan kaynakları politikaları güncellenmesi gerekiyor. Yeni düzenlemeler var.",
    "Finansal analiz raporu hazırlanması gerekiyor. Yatırım kararları için gerekli.",
    "Üretim kalite kontrolü yapılması gerekiyor. Standartlara uygunluk kontrolü.",
    "Marka kimliği çalışması yapılması gerekiyor. Logo ve kurumsal kimlik güncellemesi.",
    "Ağ güvenliği güncellemesi yapılması gerekiyor. Güvenlik açıkları tespit edildi.",
    "Çalışan performans değerlendirmesi yapılması gerekiyor. Yıllık değerlendirme süreci.",
    "Nakit akış analizi yapılması gerekiyor. Likidite durumu kontrol edilecek.",
    "Tedarik zinciri optimizasyonu yapılması gerekiyor. Maliyet düşürme çalışması.",
    "Online reklam kampanyası başlatılması gerekiyor. Google Ads optimizasyonu.",
    "Sunucu kapasitesi artırılması gerekiyor. Yüksek trafik nedeniyle yavaşlık.",
    "İşe alım süreci başlatılması gerekiyor. Yeni pozisyonlar için mülakatlar.",
    "Maliyet muhasebesi raporu hazırlanması gerekiyor. Ürün maliyet analizi.",
    "Kalite yönetim sistemi kurulması gerekiyor. ISO sertifikasyon süreci.",
    "Kurumsal iletişim stratejisi geliştirilmesi gerekiyor. Halkla ilişkiler planı.",
    "Yazılım güncellemesi yapılması gerekiyor. Yeni versiyon yayınlandı.",
    "Çalışan memnuniyet anketi yapılması gerekiyor. İş ortamı değerlendirmesi.",
    "Yatırım portföyü analizi yapılması gerekiyor. Risk değerlendirmesi gerekli.",
    "Operasyonel verimlilik artırılması gerekiyor. Süreç iyileştirme çalışması.",
    "Müşteri deneyimi iyileştirilmesi gerekiyor. Memnuniyet artırma stratejisi.",
    "Siber güvenlik eğitimi verilmesi gerekiyor. Çalışanlar için farkındalık programı.",
    "Ücretlendirme sistemi gözden geçirilmesi gerekiyor. Adil ödeme politikası.",
    "Kredi değerlendirmesi yapılması gerekiyor. Yeni müşteri risk analizi.",
    "Tedarikçi değerlendirmesi yapılması gerekiyor. Performans kriterleri kontrolü.",
    "İçerik pazarlama stratejisi geliştirilmesi gerekiyor. Blog ve sosyal medya planı.",
    "Bulut altyapısı kurulması gerekiyor. Veri merkezi taşıma projesi.",
    "Çalışan gelişim planı hazırlanması gerekiyor. Kariyer yolu belirleme.",
    "Finansal planlama yapılması gerekiyor. 5 yıllık stratejik plan.",
    "Stok yönetimi optimizasyonu yapılması gerekiyor. Envanter kontrol sistemi.",
    "Müşteri kazanım stratejisi geliştirilmesi gerekiyor. Pazarlama kanalları analizi.",
    "Veri yedekleme sistemi kurulması gerekiyor. Felaket kurtarma planı.",
    "İş süreçleri dokümantasyonu yapılması gerekiyor. Standart operasyon prosedürleri.",
    "Mali denetim yapılması gerekiyor. Bağımsız denetim firması ile çalışma.",
    "Üretim planlaması yapılması gerekiyor. Kapasite ve talep analizi.",
    "Marka farkındalığı artırılması gerekiyor. Reklam ve tanıtım kampanyası.",
    "Ağ altyapısı güçlendirilmesi gerekiyor. Fiber internet bağlantısı kurulumu.",
    "Çalışan motivasyonu artırılması gerekiyor. Takdir ve ödül sistemi.",
    "Risk yönetimi stratejisi geliştirilmesi gerekiyor. Sigorta ve güvence planı.",
    "Tedarik zinciri sürdürülebilirliği artırılması gerekiyor. Çevre dostu tedarikçiler.",
    "Dijital dönüşüm projesi başlatılması gerekiyor. Teknoloji modernizasyonu.",
    "İnsan kaynakları veri analizi yapılması gerekiyor. Çalışan istatistikleri raporu."
]

# Test verisi oluştur
data = []
for i in range(150):  # Her sınıftan en az 30 örnek olması için
    upper_class = np.random.choice(upper_classes)
    lower_class = np.random.choice(lower_classes[upper_class])
    
    # Rastgele metin seç
    text = np.random.choice(test_texts)
    summary = text[:50] + "..." if len(text) > 50 else text
    
    # Diğer alanlar
    etkilenecek_kanallar = np.random.choice(['Web', 'Mobil', 'Masaüstü', 'API', 'Tümü'])
    talep_tipi = np.random.choice(['Teknik Destek', 'Geliştirme', 'Analiz', 'Rapor', 'Eğitim'])
    talep_alt_tipi = np.random.choice(['Acil', 'Normal', 'Düşük Öncelik', 'Yüksek Öncelik'])
    reporter_birim = np.random.choice(['IT Departmanı', 'İnsan Kaynakları', 'Finans', 'Operasyon', 'Pazarlama'])
    reporter_direktorluk = np.random.choice(['Teknoloji', 'İnsan Kaynakları', 'Finans', 'Operasyon', 'Pazarlama'])
    efor = np.random.randint(1, 40)  # 1-40 saat arası
    
    data.append({
        'SUMMARY': summary,
        'description': text,
        'anaSorumluBirimUstBirim': upper_class,
        'EtkilenecekKanallar': etkilenecek_kanallar,
        'talep_tipi': talep_tipi,
        'talep_alt_tipi': talep_alt_tipi,
        'reporterBirim': reporter_birim,
        'reporterDirektorluk': reporter_direktorluk,
        'AnaSorumluBirim_Duzenlenmis': lower_class,
        'EFOR_ANASORUMLU_HARCANAN': efor
    })

# DataFrame oluştur
df = pd.DataFrame(data)

# CSV dosyasına kaydet
df.to_csv('0910.csv', sep=';', encoding='utf-8-sig', index=False)

print("✅ Test CSV dosyası oluşturuldu: 0910.csv")
print(f"📊 Toplam kayıt sayısı: {len(df)}")
print(f"📈 Üst seviye sınıflar: {df['anaSorumluBirimUstBirim'].value_counts().to_dict()}")
print(f"📉 Alt seviye sınıflar: {df['AnaSorumluBirim_Duzenlenmis'].value_counts().to_dict()}")
print("\n🎯 Dosya hazır! Şimdi 'python train_models.py' komutu ile modeli eğitebilirsiniz.")
