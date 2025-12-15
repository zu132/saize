import base64
import os
import uuid # ファイル名生成用
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from datetime import datetime

# Flaskアプリケーションの初期化
app = Flask(__name__)
# 画像をアップロードするフォルダを指定
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
# @app.route('/compare', methods=['POST'])



# 画像を一時的に保存する関数（新規作成またはリファクタ）
def save_b64_image(b64_string, upload_folder):
    """Base64文字列を受け取り、画像ファイルとして保存する"""
    try:
        # ヘッダー部分(e.g., "data:image/png;base64,")を分離
        header, encoded = b64_string.split(",", 1)
        data = base64.b64decode(encoded)
        
        # 拡張子を決定
        file_extension = header.split(';')[0].split('/')[1]
        if file_extension not in ['png', 'jpeg', 'jpg']:
            file_extension = 'png' # 不明な場合はpngに

        # ユニークなファイル名を生成
        file_name = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(upload_folder, file_name)

        with open(file_path, "wb") as f:
            f.write(data)
            
        return file_path
    except Exception as e:
        print(f"Error saving b64 image: {e}")
        return None


@app.route('/compare', methods=['POST'])
def compare():
    if request.method == 'POST':
        data = request.get_json()
        image1_b64 = data.get('image1_b64')
        image2_b64 = data.get('image2_b64')

        if not image1_b64 or not image2_b64:
            return jsonify({'error': '画像データが見つかりません'}), 400

        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        # Base64を一時ファイルとして保存
        temp_image_path1 = save_b64_image(image1_b64, upload_folder)
        temp_image_path2 = save_b64_image(image2_b64, upload_folder)
        
        if not temp_image_path1 or not temp_image_path2:
            return jsonify({'error': '画像の保存に失敗しました'}), 500

        # 既存の比較処理を呼び出し、返り値は「ファイル名」であることを明確にする
        result_filename = compare_images(temp_image_path1, temp_image_path2)
        
        # 一時ファイルを削除
        if os.path.exists(temp_image_path1):
            os.remove(temp_image_path1)
        if os.path.exists(temp_image_path2):
            os.remove(temp_image_path2)

        if result_filename:
            
            # 返された「ファイル名」を使って、フロントエンド用のURLを組み立てる
            final_image_url = f"/{upload_folder}/{result_filename}"
            print(f"フロントエンドに返す画像のURL: {final_image_url}") # デバッグ用にURLをコンソールに表示
            return jsonify({'result_image': final_image_url})
        else:
            return jsonify({'error': '比較処理中にエラーが発生しました'}), 500

    return jsonify({'error': '不正なリクエストです'}), 400
# Pythonスクリプトとして直接実行された場合にサーバーを起動
if __name__ == '__main__':
    app.run(debug=True)