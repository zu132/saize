import os
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from datetime import datetime

# Flaskアプリケーションの初期化
app = Flask(__name__)
# 画像をアップロードするフォルダを指定
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 位置合わせ処理を組み込んだ compare_images 関数
def compare_images(img_path1, img_path2):
    # --- パラメータ設定 ---
    # これらの値を調整することで、検出精度が変わります。
    MIN_MATCH_COUNT = 10      # 位置合わせに必要な最低限のマッチング数
    GOOD_MATCH_RATE = 0.7     # 良いマッチングペアとして採用する割合 (上位70%)
    ORB_FEATURES = 2000       # 検出する特徴点の最大数

    # --- 1. 画像の読み込み ---
    img1_orig = cv2.imread(img_path1)
    img2_orig = cv2.imread(img_path2)
    
    # グレースケールに変換
    img1_gray = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2GRAY)

    # --- 2. 特徴点のマッチングによる位置合わせ ---
    try:
        # ORB検出器の初期化
        orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

        # 画像1の特徴点と記述子を計算
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        # 画像2の特徴点と記述子を計算
        kp2, des2 = orb.detectAndCompute(img2_gray, None)

        # Brute-Force マッチャーを作成
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # マッチングの距離が短い順にソート
        matches = sorted(matches, key=lambda x: x.distance)

        # 上位の「良い」マッチングペアを選別
        good_matches = matches[:int(len(matches) * GOOD_MATCH_RATE)]

        if len(good_matches) > MIN_MATCH_COUNT:
            # 良いマッチングペアから、対応する特徴点の座標を取得
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # ホモグラフィ行列を計算
            h_matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            # 画像2を画像1の視点に合わせるように変形（ワープ）
            height, width = img1_gray.shape
            img2_warped = cv2.warpPerspective(img2_orig, h_matrix, (width, height))
            
            # 位置合わせ後の画像で比較処理を行う
            img_to_compare1 = img1_orig
            img_to_compare2 = img2_warped
            print("位置合わせに成功しました。")

        else:
            print(f"十分な特徴点が見つかりませんでした ({len(good_matches)}/{MIN_MATCH_COUNT})。位置合わせをスキップします。")
            # 位置合わせが失敗した場合は、リサイズのみで対応
            height, width, _ = img1_orig.shape
            img_to_compare1 = img1_orig
            img_to_compare2 = cv2.resize(img2_orig, (width, height))

    except Exception as e:
        print(f"位置合わせ中にエラーが発生: {e}")
        # エラー時もリサイズでフォールバック
        height, width, _ = img1_orig.shape
        img_to_compare1 = img1_orig
        img_to_compare2 = cv2.resize(img2_orig, (width, height))


    # --- 3. 位置合わせ後の画像で差分を検出 ---
    # ここからの処理は前回とほぼ同じ
    gray1 = cv2.cvtColor(img_to_compare1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_to_compare2, cv2.COLOR_BGR2GRAY)

    blur_gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur_gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    diff = cv2.absdiff(blur_gray1, blur_gray2)
    
    # 閾値は、位置合わせが成功していることを前提に、少し下げても良いかもしれない
    _ , thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    dilated = cv2.dilate(thresh, None, iterations=3) # 膨張処理を少し強めに

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img_to_compare1.copy()
    
    for contour in contours:
        # 面積の閾値は、検出したい間違いのサイズに合わせて調整
        if cv2.contourArea(contour) > 150: 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    result_filename = f'result_{timestamp}.png'
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, result_img)
    
    return result_filename

# --- Webページのルーティング ---

# ルートURL ('/') にアクセスがあった場合
@app.route('/', methods=['GET'])
def index():
    # index.html を表示する
    return render_template('index.html')

# '/compare' に画像がPOSTされた場合
@app.route('/compare', methods=['POST'])
def compare():
    # ファイルが2つ送られてきているかチェック
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': '画像が2枚アップロードされていません'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    # ファイル名を取得して保存
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
    file1.save(path1)
    file2.save(path2)
    
    # 画像比較関数を呼び出す
    try:
        result_image_name = compare_images(path1, path2)
        # 成功したら結果画像のURLを返す
        return jsonify({'result_image': f'/{UPLOAD_FOLDER}/{result_image_name}'})
    except Exception as e:
        return jsonify({'error': f'処理中にエラーが発生しました: {e}'}), 500

# Pythonスクリプトとして直接実行された場合にサーバーを起動
if __name__ == '__main__':
    app.run(debug=True)