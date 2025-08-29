import time     #時間計測をできるようにする

start_time = time.time()

# 計測したいコード
result = sum(range(1000000))

end_time = time.time()
print(f"実行時間: {end_time - start_time}秒")


from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split    #scikit-learnライブラリのデータセットをトレーニングセットとテストセットに分割する
import matplotlib.pyplot as plt #データをグラフ作成などで可視化できる
import pickle   #オブジェクトを変換し保存したり、ファイルを読み込んでオブジェクトを復元したりできる
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers



CLASS = 4
TEST_IMG_NUM = 15   
IMAGE_SIZE = (256,256)  #画像サイズ
IMG_DIR = 'img/'        #画像dir path
IMG_ORIGINAL_DIR = f'{IMG_DIR}original/'    #original（元画像） dir path
IMG_STD_DIR = f'{IMG_DIR}output/'   #出力dir path
IMG_VALIDATION_DIR = f'{IMG_DIR}validation/'    #validation path
IMG_TEST_DIR = f'{IMG_DIR}test/'    #test dir path
CONFIRM_IMG_DIR = f'{IMG_DIR}confirmation/'     #confirmation path
TARGET_DIR_LIST = os.listdir(IMG_ORIGINAL_DIR)  #元画像dir内のリスト
model_filename = f'model/face_recog_model_{TARGET_DIR_LIST}.h5'   #モデル名


############## 訓練データ ##############
# ファイルパス取得
file_path_array = []        #配列定義
for target in TARGET_DIR_LIST:      #元画像リストループ
    file_path = f'{IMG_STD_DIR}{target}/*.jpg'  #文字列結合しファイルパス作成
    print(file_path)
    file_path_array.append(glob.glob(file_path))      #条件に合致するファイルパスリストをリストの中に追加
# 画像読み込み
img_dataset = {}    #連想配列みたいなやつ 辞書
data = []       #data配列宣言
target = []     #target配列宣言
img_dataset['data'] = data      #辞書に配列追加
img_dataset['target'] = target  #辞書に配列追加
for dir_num, target_dir in enumerate(TARGET_DIR_LIST):  #元画像リストループ
    for img_file_path in file_path_array[dir_num]:      #元画像リストのファイルパスリストをループ
        img = load_img(img_file_path, color_mode = 'rgb', target_size=IMAGE_SIZE)   #ファイルを読み込み指定した形式で返す   画像ファイルのパス,カラーモード,サイズ
        img_list = img_to_array(img)    #画像の情報をNumPyリストに格納
        img_dataset['data'].append(img_list)    #リストを辞書に格納
        img_dataset['target'].append(dir_num)   #ラベルを辞書に格納

img_X_train = np.array(img_dataset['data'])     #画像データの配列をNumpy配列へ変換
img_Y_train = np.array(img_dataset['target'])   #ラベルデータの配列をNumpy配列に変換


############## 検証データ ##############
# ファイルパス取得
file_path_array = []        #配列定義
for target in TARGET_DIR_LIST:      #元画像リストループ
    file_path = f'{IMG_VALIDATION_DIR}{target}/*.jpg'  #文字列結合しファイルパス作成
    # print(file_path)
    file_path_array.append(glob.glob(file_path))     #条件に合致するファイルパスリストをリストの中に追加

# 画像読み込み
img_dataset = {}        #連想配列みたいなやつ 辞書
data = []               #data配列宣言
target = []             #target配列宣言
img_dataset['data'] = data      #辞書に配列追加
img_dataset['target'] = target  #辞書に配列追加
for dir_num, target_dir in enumerate(TARGET_DIR_LIST):  #元画像リストループ
    for img_file_path in file_path_array[dir_num]:        #元画像リストのファイルパスリストをループ
        img = load_img(img_file_path, color_mode = 'rgb', target_size=IMAGE_SIZE)   #ファイルを読み込み指定した形式で返す   画像ファイルのパス,カラーモード,サイズ
        img_list = img_to_array(img)        #画像の情報をNumPyリストに格納
        img_dataset['data'].append(img_list)    #リストを辞書に格納
        img_dataset['target'].append(dir_num)   #ラベルを辞書に格納

img_X_valid = np.array(img_dataset['data'])     #画像データの配列をNumpy配列へ変換
img_Y_valid = np.array(img_dataset['target'])   #ラベルデータの配列をNumpy配列に変換


############## テストデータ ##############
# ファイルパス取得
file_path_array = []    #ファイルパス配列作成
for target in TARGET_DIR_LIST:      #元画像ループ
    file_path = f'{IMG_TEST_DIR}{target}/*.jpg'     #元画像path作成
    # print(file_path)
    file_path_array.append(glob.glob(file_path))    #元画像pathを配列へ

# 画像読み込み
img_dataset = {}    #辞書宣言
data = []       #配列宣言
target = []     #配列宣言
img_dataset['data'] = data      #辞書に配列追加
img_dataset['target'] = target  #辞書に配列追加
for dir_num, target_dir in enumerate(TARGET_DIR_LIST):  #元画像ループ
    for img_file_path in file_path_array[dir_num]:      #元画像path配列をループ
        img = load_img(img_file_path, color_mode = 'rgb', target_size=IMAGE_SIZE)   #ファイルを読み込み指定した形式で返す   画像ファイルのパス,カラーモード,サイズ
        img_list = img_to_array(img)        #画像の情報をNumPyリストに格納
        img_dataset['data'].append(img_list)    #リストを辞書に格納
        img_dataset['target'].append(dir_num)   #ラベルを辞書に格納
        

img_X_test = np.array(img_dataset['data'])      #画像データの配列をNumpy配列へ変換
img_Y_test = np.array(img_dataset['target'])    #ラベルデータの配列をNumpy配列に変換

img_X_train = np.array(img_X_train)     #NumPy配列に変換
img_X_valid = np.array(img_X_valid)     #NumPy配列に変換
img_X_test = np.array(img_X_test)       #NumPy配列に変換

###モデル構築学習###

# データの標準化
img_X_train = img_X_train.astype('float32') / 255.
img_X_valid = img_X_valid.astype('float32') / 255.
img_X_test = img_X_test.astype('float32') / 255.

# one-hotベクトル化
img_Y_train = to_categorical(img_Y_train, num_classes=CLASS)  # num_classes はクラス数に合わせる
img_Y_valid = to_categorical(img_Y_valid, num_classes=CLASS)
img_Y_test = to_categorical(img_Y_test, num_classes=CLASS)

#エンコーディング形式に変換 
img_Y_train = img_Y_train[:len(img_X_train)]
img_Y_valid = img_Y_valid[:len(img_X_valid)]
img_Y_test = img_Y_test[:len(img_X_test)]

# 転移学習のベースモデルとしてVGG16を宣言
input_tensor = Input(shape=(256, 256, 3))   #モデルの入力データの形状指定   高さ、幅、チャンネル数
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)     #ニューラルネットワークアーキテクチャの1つで画像分類に使用される

# VGG16以降のモデル作成
top_model = Sequential()

top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))  # VGG16の出力を1次元に平坦化
top_model.add(Dense(256, activation='relu'))        # 全結合層1（256ユニット、ReLU活性化関数）
top_model.add(Dropout(0.2))         # ドロップアウト（20%のユニットをランダムに無効化）
top_model.add(Dense(128, activation='relu'))        # 全結合層2（128ユニット、ReLU活性化関数）
top_model.add(Dropout(0.2))         # ドロップアウト（20%のユニットをランダムに無効化）
top_model.add(Dense(64, activation='relu'))         # 全結合層3（64ユニット、ReLU活性化関数）
top_model.add(Dropout(0.2))     # ドロップアウト（20%のユニットをランダムに無効化）

top_model.add(Dense(CLASS, activation='softmax'))       # 出力層（5クラス分類、Softmax活性化関数）

# VGG16と新規モデルの連結
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# 重み付け値の固定
for layer in model.layers[:19]:
    layer.trainable = False


# モデル構築
model.compile(loss='categorical_crossentropy',      #モデルのコンパイル
            optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
            run_eagerly=False,      #デバック環境ならTrue
            metrics=['accuracy'])

# モデル学習
history = model.fit(        #モデルトレーニングを行うメソッド
    img_X_train, img_Y_train,
    validation_data=(img_X_valid, img_Y_valid),
    batch_size=32,
    epochs=5)

model.save(model_filename)      #モデル保存 



# 学習状況
## 訓練と検証
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

## 正解率
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

## 損失関数
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 10.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# 精度の評価
scores = model.evaluate(img_X_test, img_Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
