from mediapipe import solutions 
from mediapipe.framework.formats import landmark_pb2 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# Eğitilmiş modeli yükle
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"'{MODEL_PATH}' modeli başarıyla yüklendi.")
except FileNotFoundError:
    print(f"Hata: '{MODEL_PATH}' modeli bulunamadı. Lütfen önce egitim.py ile modeli eğitin.")
    exit()
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    exit()

# MediaPipe FaceLandmarker nesnesi oluştur
try:
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=False,
                                           num_faces=1) # Tek yüz varsayımı
    detector = vision.FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"FaceLandmarker başlatılırken hata: {e}")
    print("Lütfen 'face_landmarker_v2_with_blendshapes.task' model dosyasının doğru konumda olduğundan emin olun.")
    exit()


def process_landmarks_for_prediction(rgb_image, detection_result, trained_model):
    annotated_image = np.copy(rgb_image)
    predicted_expression = "BİLİNMİYOR" # Varsayılan etiket

    if detection_result.face_landmarks:
        face_landmarks_list = detection_result.face_landmarks
        if face_landmarks_list:
            face_landmarks = face_landmarks_list[0] 

        
            landmark_coordinates = []
            if len(face_landmarks) == 478:
                for landmark in face_landmarks:
                    landmark_coordinates.append(landmark.x)
                    landmark_coordinates.append(landmark.y)
                
              
                try:
                    prediction = trained_model.predict([landmark_coordinates])
                    predicted_expression = prediction[0].upper()
                except Exception as e:
                    print(f"Tahmin sırasında hata: {e}")
                    predicted_expression = "HATA"

            else:
                predicted_expression = "LANDMARK SAYISI UYUŞMUYOR"
        
    # Tahmin edilen ifadeyi ekrana yazdır
    cv2.putText(annotated_image, 
                predicted_expression, 
                (30, 60), # Metin konumu
                cv2.FONT_HERSHEY_SIMPLEX, # Font tipi
                1.5, # Font ölçeği
                (0, 255, 0) if predicted_expression not in ["BİLİNMİYOR", "HATA", "LANDMARK SAYISI UYUŞMUYOR"] else (0,0,255), # Yeşil renk, hata ise kırmızı
                3) # Kalınlık
    

    return annotated_image

# Kameradan görüntü alımı
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Kamera açılamadı!")
    exit()

print("\nGerçek Zamanlı Yüz İfadesi Tanıma Başlatıldı...")
print("Çıkmak için 'q' tuşuna basın.")

while cam.isOpened():
    basari, frame = cam.read()
    if not basari:
        print("Kameradan görüntü alınamadı.")
        break

    # Görüntüyü RGB'ye çevir
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Yüz ilgin noktalarını tespit et
    detection_result = detector.detect(mp_image)

    # Landmarkları işle, tahmin yap ve ekranda göster
    output_frame_rgb = process_landmarks_for_prediction(rgb_frame, detection_result, model)
    
    # Görüntüyü tekrar BGR'ye çevirerek OpenCV ile göster
    output_frame_bgr = cv2.cvtColor(output_frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Gerçek Zamanlı Yüz İfadesi Tanıma", output_frame_bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Uygulama sonlandırıldı.")