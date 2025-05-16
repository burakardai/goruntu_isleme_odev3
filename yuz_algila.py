from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os 

# Hedef ifadeler ve her ifade için toplanacak minimum örnek sayısı
EXPRESSIONS = ["mutlu", "uzgun", "kizgin", "saskin"]
MIN_SAMPLES_PER_EXPRESSION = 200
CSV_FILE = "veriseti.csv"

# Global değişkenler
current_expression_index = 0
sample_counts = {expr: 0 for expr in EXPRESSIONS}
header_written = False

def sutun_basliklarini_olustur(dosya_adi):
    global header_written
    # Dosya yoksa veya boşsa başlıkları yaz
    if not os.path.exists(dosya_adi) or os.path.getsize(dosya_adi) == 0:
        with open(dosya_adi, "w") as f:
            satir = ""
            # MediaPipe FaceLandmarker 478 landmark döndürür (468 yüz + 10 iris)
            # Her landmark için x ve y koordinatları
            for i in range(1, 479): # 1'den 478'e kadar
                satir += f"x{i},y{i},"
            satir += "Etiket\n"
            f.write(satir)
        print(f"'{dosya_adi}' için başlıklar oluşturuldu.")
        header_written = True
    else:
        print(f"'{dosya_adi}' zaten mevcut ve başlıklar var gibi görünüyor.")
        header_written = True


def process_and_save_landmarks(rgb_image, detection_result, current_expression_label):
    global sample_counts
    
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    if not face_landmarks_list:
        return annotated_image # Yüz bulunamazsa orijinal görüntüyü döndür

    # num_faces=1 olduğu için sadece ilk yüzü alıyoruz
    face_landmarks = face_landmarks_list[0]

    # Landmark koordinatlarını (x,y) al ve CSV için hazırla
    landmark_coordinates = []
    if len(face_landmarks) == 478: # Beklenen sayıda landmark varsa
        for landmark in face_landmarks:
            landmark_coordinates.append(str(round(landmark.x, 5)))
            landmark_coordinates.append(str(round(landmark.y, 5)))
        
        row_data = ",".join(landmark_coordinates)
        row_data += f",{current_expression_label}\n"
        
        with open(CSV_FILE, "a") as f:
            f.write(row_data)
        
        sample_counts[current_expression_label] += 1
        print(f"'{current_expression_label}' için {sample_counts[current_expression_label]}. örnek kaydedildi.")
    else:
        print(f"Uyarı: Beklenen 478 landmark yerine {len(face_landmarks)} landmark bulundu. Veri kaydedilmedi.")



    return annotated_image


try:
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1) # Tek yüz varsayımı
    detector = vision.FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"FaceLandmarker başlatılırken hata: {e}")
    print("Lütfen 'face_landmarker_v2_with_blendshapes.task' model dosyasının doğru konumda olduğundan emin olun.")
    exit()

# CSV başlıklarını oluştur/kontrol et
sutun_basliklarini_olustur(CSV_FILE)

# Kameradan görüntü alımı
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Kamera açılamadı!")
    exit()

print("\nVeri Toplama Başlıyor...")
print("------------------------------------")
print(f"İfadeler: {', '.join(EXPRESSIONS)}")
print(f"Her ifade için hedef örnek sayısı: {MIN_SAMPLES_PER_EXPRESSION}")
print("------------------------------------")

while cam.isOpened():
    current_expression = EXPRESSIONS[current_expression_index]
    
    basari, frame = cam.read()
    if not basari:
        print("Kameradan görüntü alınamadı.")
        break

    # Görüntüyü RGB'ye çevir
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # STEP 4: Yüz ilgin noktalarını tespit et
    detection_result = detector.detect(mp_image)
    

    
    display_frame = np.copy(frame)

    # Kullanıcıya bilgi mesajları
    cv2.putText(display_frame, f"IFADE: {current_expression.upper()}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"ORNEK: {sample_counts[current_expression]}/{MIN_SAMPLES_PER_EXPRESSION}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    info_text = "'S': Kaydet | 'N': Sonraki Ifade | 'Q': Kapat"
    if sample_counts[current_expression] >= MIN_SAMPLES_PER_EXPRESSION:
         cv2.putText(display_frame, "Bu ifade için yeterli örnek toplandı.", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(display_frame, info_text, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Veri Toplama - Yuz Algılama", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("Veri toplama sonlandırılıyor...")
        break
    elif key == ord('s') or key == ord('S'):
        if detection_result and detection_result.face_landmarks:
            
            face_landmarks_list_for_save = detection_result.face_landmarks
            if face_landmarks_list_for_save:
                face_landmarks_for_save = face_landmarks_list_for_save[0]
                landmark_coordinates = []
                if len(face_landmarks_for_save) == 478:
                    for landmark in face_landmarks_for_save:
                        landmark_coordinates.append(str(round(landmark.x, 5)))
                        landmark_coordinates.append(str(round(landmark.y, 5)))
                    
                    row_data = ",".join(landmark_coordinates)
                    row_data += f",{current_expression}\n"
                    
                    with open(CSV_FILE, "a") as f:
                        f.write(row_data)
                    
                    sample_counts[current_expression] += 1
                    print(f"'{current_expression}' için {sample_counts[current_expression]}. örnek kaydedildi.")
                else:
                    print(f"Uyarı: Kayıt için beklenen 478 landmark yerine {len(face_landmarks_for_save)} landmark bulundu. Veri kaydedilmedi.")
            else:
                 print("Yüz bulunamadı, 's' tuşuna basıldı ama veri kaydedilemedi.")
        else:
            print("Yüz bulunamadı, 's' tuşuna basıldı ama veri kaydedilemedi.")

    elif key == ord('n') or key == ord('N'):
        current_expression_index = (current_expression_index + 1)
        if current_expression_index >= len(EXPRESSIONS):
            current_expression_index = 0 # Başa dön veya tamamlandığını belirt
            print("\nTüm ifadeler bir tur döndü. İsterseniz devam edebilir veya 'q' ile çıkabilirsiniz.")
            # Tüm ifadeler için yeterli örnek toplandı mı kontrolü
            all_done = True
            for expr in EXPRESSIONS:
                if sample_counts[expr] < MIN_SAMPLES_PER_EXPRESSION:
                    all_done = False
                    print(f"'{expr}' icin hala ornek gerekiyor ({sample_counts[expr]}/{MIN_SAMPLES_PER_EXPRESSION}).")
            if all_done:
                print("\nTEBRİKLER! Tüm ifadeler için yeterli sayıda örnek toplandı!")
                print("Çıkmak için 'q' tuşuna basabilirsiniz.")
        
        print(f"Aktif ifade: {EXPRESSIONS[current_expression_index].upper()}")


cam.release()
cv2.destroyAllWindows()
print("Veri toplama tamamlandı.")
print("Toplanan örnek sayıları:")
for expr, count in sample_counts.items():
    print(f"- {expr}: {count}")