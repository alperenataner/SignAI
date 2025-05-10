# SignAI
# İşaret Dili Tanıma Sistemi - Teknik Rapor

## 1. Proje Özeti

Bu proje, gerçek zamanlı işaret dili tanıma sistemini web tabanlı bir arayüz üzerinden sunmayı amaçlamaktadır. Sistem, kullanıcının kamera görüntüsünü analiz ederek işaret dilindeki harfleri tanımlamakta ve sonuçları anlık olarak göstermektedir.

## 2. Sistem Mimarisi

### 2.1 Bileşenler
- **Web Arayüzü**: Flask tabanlı web uygulaması
- **Görüntü İşleme**: OpenCV kütüphanesi
- **Derin Öğrenme Modeli**: PyTorch tabanlı CNN modeli
- **Donanım Desteği**: CUDA GPU desteği (opsiyonel)

### 2.2 Teknoloji Yığını
- **Backend**: Python 3.10
- **Web Framework**: Flask 2.3.3
- **Görüntü İşleme**: OpenCV 4.8.0
- **Derin Öğrenme**: PyTorch 2.0.1
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Framework**: Bootstrap 5.3.0

## 3. Sistem Akışı

### 3.1 Veri Akışı
1. Kullanıcı web arayüzüne erişir
2. Kamera görüntüsü Flask sunucusuna iletilir
3. Görüntü işleme ve model tahmini yapılır
4. Sonuçlar gerçek zamanlı olarak kullanıcıya gösterilir

### 3.2 Görüntü İşleme Süreci
1. Kamera görüntüsü alınır
2. Görüntü gri tonlamaya dönüştürülür
3. Boyut 64x64 piksele ayarlanır
4. Tensor formatına dönüştürülür
5. Model tahmini yapılır
6. Sonuçlar görüntü üzerine yazılır

## 4. Model Mimarisi

### 4.1 CNN Modeli
- 4 konvolüsyon katmanı
- Batch Normalization
- ReLU aktivasyon fonksiyonu
- Max Pooling
- Dropout (0.5)
- 29 sınıf çıkışı (26 harf + 'space' + 'del' + 'nothing')

### 4.2 Model Özellikleri
- Giriş boyutu: 64x64x1 (gri tonlamalı)
- Çıkış boyutu: 29 (sınıf sayısı)
- Optimizer: Adam
- Loss Function: Cross Entropy

## 5. Performans ve Optimizasyon

### 5.1 Performans İyileştirmeleri
- CUDA GPU desteği
- Batch işleme
- Görüntü ön işleme optimizasyonu
- Model quantizasyonu

### 5.2 Hata Yönetimi
- Kamera bağlantı kontrolü
- Model yükleme kontrolü
- Görüntü işleme hata yakalama
- Kullanıcı geri bildirimi

## 6. Güvenlik Önlemleri

### 6.1 Uygulama Güvenliği
- Flask güvenlik başlıkları
- CORS politikaları
- Input validasyonu
- Hata mesajlarının gizlenmesi

### 6.2 Veri Güvenliği
- Yerel işleme
- Veri şifreleme
- Güvenli bağlantı (HTTPS)

## 7. Kullanım Kılavuzu

### 7.1 Sistem Gereksinimleri
- Python 3.10 veya üzeri
- Web kamerası
- NVIDIA GPU (opsiyonel)
- 4GB RAM minimum

### 7.2 Kurulum
```bash
pip install -r requirements.txt
python app.py
```

### 7.3 Kullanım
1. Web tarayıcısında http://localhost:5000 adresine gidin
2. Kamera izinlerini verin
3. İşaret dilini kameraya gösterin
4. Sonuçları ekranda görüntüleyin

## 8. Gelecek Geliştirmeler

### 8.1 Planlanan İyileştirmeler
- Kelime ve cümle tanıma
- Farklı işaret dilleri desteği
- Mobil uygulama
- Offline mod desteği

### 8.2 Araştırma Alanları
- Daha hızlı model mimarileri
- Daha iyi ön işleme teknikleri
- Transfer öğrenme uygulamaları

## 9. Sonuç

Bu proje, işaret dili tanıma teknolojisini web tabanlı bir platform üzerinden erişilebilir hale getirmeyi başarmıştır. Gerçek zamanlı işleme, kullanıcı dostu arayüz ve güvenilir performans ile kullanıcılara pratik bir çözüm sunmaktadır.

## 10. Referanslar

- PyTorch Documentation
- OpenCV Documentation
- Flask Documentation
- Bootstrap Documentation 
