from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os, send2trash
import glob
import shutil

#元画像パス
img_dir_path = 'img/'

#サイズ変更後画像パス
original_dir_path = img_dir_path + 'original/'

#dir一覧取得
IMG_LABEL_LIST = os.listdir(original_dir_path)

#リサイズ画像パス
resized_original_path = img_dir_path + 'resized/'

#出力画像パス
opencv_img_dir_path = img_dir_path + 'output/'

#変更後の画像幅
width_resize = 200 

for dir_idx, dir_name in enumerate(IMG_LABEL_LIST): #enumerateはリストにインデックスを付ける関数
    input_path = f'{original_dir_path}{dir_name}' # オリジナル画像のパスリスト
    files = glob.glob(input_path + '/*.jpg')    #全てのjpgをリスト化
    print(files)
    files = [x.replace("\\","/") for x in files]
    output_path = f'{resized_original_path}{dir_name}' # リサイズ後の出力先のパスリスト

    
    #既存のdirがあった場合、削除し生成
    if os.path.isdir(output_path) == True:
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    
    for file_idx, file_name in enumerate(files):
        
        img = cv2.imread(file_name)  # 元画像の読み出し
        height, width, ch = img.shape  # 画像の高さ、幅、チャンネル数の取得
        height_resize = int(height*width_resize/width)  # 変更後画像高さの計算
        img_resize = cv2.resize(img,dsize=(width_resize,height_resize))  #画像サイズの変更 
        cv2.imwrite(f'{resized_original_path}{dir_name}/{str(dir_idx).zfill(2)}_{str(file_idx).zfill(3)}.jpg',img_resize)  # 画像の保存  
        
# 画像変換条件
# 収縮・膨張用フィルタ
filter1 = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]], np.uint8)
filter2 = np.ones((3, 3))

# 総計48パターン 
#list_resize = [2, 3, 5, 7] #1. 1/2, 1/3, 1/5, 1/7にリサイズ = 4パターン
#list_mosaic = [3, 5, 7, 10] #2. リサイズを元に戻すことでモザイク処理 = 4パターン
list_rotation = [30, 45, 60, 90] #3. 画像の回転 = 4パターン
list_flip = [0, 1, -1] #4. 左右・上下・軸反転 = 3パターン
list_cvt1 = [0] #5. BGRからLab色空間に変換 = 1パターン
list_cvt2 = [0] #6. BGRからモノクロ画像に変換 = 1パターン
list_cvt3 = [0] #7. 色反転 = 1パターン
# 閾値処理4 × 5 = 20パターン
list_THRESH_BINARY = [50, 100, 150, 200] #8.
list_THRESH_BINARY_INV = [50, 100, 150, 200] #9.
list_THRESH_TRUNC = [50, 100, 150, 200] #10.
list_THRESH_TOZERO = [50, 100, 150, 200] #11.
list_THRESH_TOZERO_INV = [50, 100, 150, 200] #12.
list_gauss = [11, 31, 51, 71] #13. ぼかし処理 = 4パターン
list_nois_gray = [0] #14. ノイズ除去（モノクロ） = 1パターン
list_nois_color = [0] #15. ノイズ除去（カラー） = 1パターン
# 2（収縮・膨張） × 2（四方・周囲８） = 4パターン
list_dilate = [filter1, filter2] #16.
list_erode = [filter1, filter2] #17.

#画像パターンをリストに全て格納
parameters = [list_rotation, list_flip, list_cvt1, list_cvt2, list_cvt3, list_THRESH_BINARY, \
            list_THRESH_BINARY_INV, list_THRESH_TRUNC, list_THRESH_TOZERO, list_THRESH_TOZERO_INV, list_gauss, \
            list_nois_gray, list_nois_color, list_dilate, list_erode]

#水増し画像の合計枚数出力
list_sum =  len(list_rotation) + len(list_flip) + len(list_cvt1) + len(list_cvt2) + len(list_cvt3) + \
            len(list_THRESH_BINARY) + len(list_THRESH_BINARY_INV) + len(list_THRESH_TRUNC) + len(list_THRESH_TOZERO) + \
            len(list_THRESH_TOZERO_INV) + len(list_gauss) + len(list_nois_gray) + len(list_nois_color) + \
            len(list_dilate) + len(list_erode)
print("合計：{}枚".format(list_sum))

#実行する関数のリスト
methods = np.array([#lambda img, i: cv2.resize(img, (img.shape[1] // i, img.shape[0] // i)), #1.
                    #lambda img, i: cv2.resize(cv2.resize(img, (img.shape[1] // i, img.shape[0] // i)), (img.shape[1],img.shape[0])), #2.
                    lambda img, i: cv2.warpAffine(img, cv2.getRotationMatrix2D(tuple(np.array([img.shape[1] / 2, img.shape[0] /2])), i, 1), (img.shape[1], img.shape[0])), #3.
                    lambda img, i: cv2.flip(img, i), #4.
                    lambda img, i: cv2.cvtColor(img, cv2.COLOR_BGR2LAB), #5.
                    lambda img, i: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), #6.
                    lambda img, i: cv2.bitwise_not(img), #7.
                    lambda img, i: cv2.threshold(img, i, 255, cv2.THRESH_BINARY)[1], #8.
                    lambda img, i: cv2.threshold(img, i, 255, cv2.THRESH_BINARY_INV)[1], #9.
                    lambda img, i: cv2.threshold(img, i, 255, cv2.THRESH_TRUNC)[1], #10.
                    lambda img, i: cv2.threshold(img, i, 255, cv2.THRESH_TOZERO)[1], #11.
                    lambda img, i: cv2.threshold(img, i, 255, cv2.THRESH_TOZERO_INV)[1], #12.
                    lambda img, i: cv2.GaussianBlur(img, (i, i), 0), #13.
                    lambda img, i: cv2.fastNlMeansDenoising(img, i), #14
                    lambda img, i: cv2.fastNlMeansDenoisingColored(img), #15.
                    lambda img, i: cv2.dilate(img, i), #16.
                    lambda img, i: cv2.erode(img, i) #17.
                    ])


#水増し画像の保存用関数
def save(img, file_path):
    cv2.imwrite(file_path, img)
    
    
#画像の水増しと保存
for dir_idx, dir_name in enumerate(IMG_LABEL_LIST):
    #入力画像の保存先パス
    input_path = f'{resized_original_path}{dir_name}'
    #ファイル名結合
    files = glob.glob(input_path + '/*.jpg')
    print(input_path)
    print(files)
    files = [x.replace("\\","/") for x in files]
    
    # 出力画像の保存先パス
    output_path = f'{opencv_img_dir_path}{dir_name}'
    # ディレクトリの初期化
    if os.path.exists(output_path): #dirの有無
        file_amount = os.listdir(output_path)   #ファイルリスト取得
        if len(file_amount) != 0:   #ファイルが存在する場合
            for del_file in os.listdir(output_path):    #削除するファイルリスト取得
                if del_file.endswith('/.jpg'):   #.jpgで終わるなら
                    send2trash.send2trash(f'{output_path}/{del_file}')  #ゴミ箱へ移動
    else:
        os.mkdir(output_path)   #dir作成
    
    for file_idx, file in enumerate(files): #filesリストをループ
        file_img = cv2.imread(file)     #画像の読み出し
        for method_idx, method in enumerate(methods):   #画像の高さ、幅、チャンネル数の取得
            for pararm_idx, param in enumerate(parameters[method_idx]):  #画像パターンループ   
                cnv_img = method(file_img, param)   #変換した画像を生成
                output_file_path = f'{output_path}/{dir_idx}_{str(file_idx).zfill(3)}_{str(method_idx + 1).zfill(3)}_{str(pararm_idx).zfill(3)}.jpg'    #ファイル名の結合
                save(cnv_img, output_file_path)     #画像の保存
    
# ファイル数確認
output_path = opencv_img_dir_path
for dir_name in IMG_LABEL_LIST:
    file_path = f'{output_path}{dir_name}/*.jpg'
    file_list = glob.glob(file_path)
    print(f'{dir_name}: {len(file_list)}')