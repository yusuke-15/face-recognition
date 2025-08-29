from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
import glob

# モデルの読み込み
model_filename = "model/face_recog_model_['furube', 'kamada', 'kato', 'kikuchi'].h5"
model = load_model(model_filename)

# クラスラベルの設定
class_labels = ['furube', 'kamada', 'kato', 'kikuchi']  # クラスラベル

# 画像ディレクトリの設定
confirmation_dir = 'img/confirmation/'  # 確認用の画像が格納されているディレクトリ

# ディレクトリ直下の全てのフォルダにある画像ファイルのパスを取得
image_paths = glob.glob(os.path.join(confirmation_dir, '*/*.jpg'))

# 画像ごとに予測を行う
for image_path in image_paths:
    # 画像の読み込みと前処理
    image = load_img(image_path, color_mode='rgb', target_size=(256, 256))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # バッチ次元を追加
    image_array = image_array.astype('float32') / 255.0  # 正規化

    # 予測
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)  # 予測クラスの取得

    # 予測結果の表示
    print(f'Image: {image_path} | Predicted class: {class_labels[predicted_class[0]]}')