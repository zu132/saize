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

# compare_images 関数全体をこちらに置き換えてください
def compare_images(img_path1, img_path2):
    # --- パラメータ設定 ---
    # ★ オープニング後に軽く膨張させるかどうかのスイッチ
    DILATE_AFTER_OPENING = True

    # モルフォロジー演算
    OPENING_KERNEL_SIZE = 3 
    OPENING_ITERATIONS = 1  # 弱めに設定

    # 面積フィルター
    MIN_CONTOUR_AREA = 40 # 検出率を上げるため、少しだけ閾値を下げてみる

    # 形フィルター
    ASPECT_RATIO_THRESHOLD = 8.0
    SOLIDITY_THRESHOLD = 0.35 # 検出率を上げるため、少しだけ閾値を下げてみる

    # 元のパラメータ
    MIN_MATCH_COUNT = 10
    GOOD_MATCH_RATE = 0.7
    ORB_FEATURES = 2000
    RANSAC_REPROJ_THRESHOLD = 3.0
    DIFF_THRESHOLD = 30

    # --- 1. 画像の読み込み ---
    img1_orig = cv2.imread(img_path1)
    img2_orig = cv2.imread(img_path2)
    img1_gray = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2GRAY)

    # --- 2. 特徴点のマッチングによる位置合わせ ---
    try:
        # (省略: この部分は変更ありません)
        orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * GOOD_MATCH_RATE)]

        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            h_matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
            height, width = img1_gray.shape
            img2_warped = cv2.warpPerspective(img2_orig, h_matrix, (width, height))
            img_to_compare1 = img1_orig
            img_to_compare2 = img2_warped
            print("位置合わせに成功しました。")
        else:
            print(f"十分な特徴点が見つかりませんでした ({len(good_matches)}/{MIN_MATCH_COUNT})。位置合わせをスキップします。")
            height, width, _ = img1_orig.shape
            img_to_compare1 = img1_orig
            img_to_compare2 = cv2.resize(img2_orig, (width, height))
    except Exception as e:
        print(f"位置合わせ中にエラーが発生: {e}")
        height, width, _ = img1_orig.shape
        img_to_compare1 = img1_orig
        img_to_compare2 = cv2.resize(img2_orig, (width, height))

    # --- 3. 位置合わせ後の画像で差分を検出 ---
    # (省略: この部分は変更ありません)
    gray1 = cv2.cvtColor(img_to_compare1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_to_compare2, cv2.COLOR_BGR2GRAY)
    blur_gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur_gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    diff_gray = cv2.absdiff(blur_gray1, blur_gray2)
    _ , thresh_gray = cv2.threshold(diff_gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    blur_color1 = cv2.GaussianBlur(img_to_compare1, (5, 5), 0)
    blur_color2 = cv2.GaussianBlur(img_to_compare2, (5, 5), 0)
    diff_color = cv2.absdiff(blur_color1, blur_color2)
    diff_color_sum = np.sum(diff_color, axis=2).astype('uint8')
    _ , thresh_color = cv2.threshold(diff_color_sum, DIFF_THRESHOLD + 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_or(thresh_gray, thresh_color)

    # ★★★ ここが重要な変更点 ★★★
    # 1. オープニング処理でノイズを除去
    kernel = np.ones((OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE), np.uint8)
    processed_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=OPENING_ITERATIONS)

    # 2. (スイッチがTrueなら) 軽く膨張させて、消えかけた間違いを補強
    if DILATE_AFTER_OPENING:
        processed_mask = cv2.dilate(processed_mask, None, iterations=1)

    # --- 4. 輪郭の検出とフィルタリング ---
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img_to_compare1.copy()
    
    # (forループによるフィルタリング処理は変更なし)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0: continue
        aspect_ratio = float(w) / h
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area / hull_area
        if aspect_ratio > ASPECT_RATIO_THRESHOLD or aspect_ratio < (1 / ASPECT_RATIO_THRESHOLD):
            continue
        if solidity < SOLIDITY_THRESHOLD:
            continue
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # --- 5. 結果の保存 ---
    # (省略: この部分は変更ありません)
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