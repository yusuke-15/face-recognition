import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model


model_filename = "model/face_recog_model_['furube', 'kamada', 'kato', 'kikuchi'].h5"

# カスケード分類器の指定
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#顔認証モデルロード
loaded_model =  load_model(model_filename)


# カメラ定義
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

while(True):
    # ビデオキャプチャ
    ret, frame = cap.read()
    
    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 顔検出
    faces = cascade.detectMultiScale(gray, scaleFactor=1.23, minNeighbors=4, minSize=(50, 50))
    
    for x, y, w, h in faces:
        face_roi = frame[y:y+h, x:x+w]      # 顔領域を切り出し
        face_resized = cv2.resize(face_roi, (256, 256))  # モデルが期待するサイズにリサイズ
        face_resized = face_resized.astype(np.float32) / 255.0  # 正規化（0-1の範囲）
        
        
        # 顔認識
        result = loaded_model.predict(np.expand_dims(face_resized, axis=0))


        print("Prediction result:", result)
        class_names = ['furube', 'kamada', 'kato', 'kikuchi']
        confidence = np.argmax(result)  # 最も高い確率
        predicted_class = class_names[confidence]
        
        #結果が一つかどうか
        
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), (127, 255,0), 2)   #矩形描画   対象画像,矩形開始位置,矩形終了位置,色,ピクセル数
        
        #戻り値顔のID,信頼度
        text = f"{predicted_class}"  #'.2f'はfloatで小数第二位まで表示
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)    #画像にテキストを描画
        
        
        
    # 表示
    cv2.imshow('frame', frame)
    # qが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# カメラ終了
cap.release()
# ウィンドウ終了
cv2.destroyAllWindows()