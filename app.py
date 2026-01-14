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

def auto_rectify(image):
    """画像から最大の四角形（台形）を検出し、長方形に補正（オートクロップ）する"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # 左上
        rect[2] = pts[np.argmax(s)] # 右下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # 右上
        rect[3] = pts[np.argmax(diff)] # 左下

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return image

def compare_images(img_path1, img_path2, use_auto_crop=False):
    """2枚の画像を比較して間違いを抽出する"""
    # パラメータ設定
    DILATE_AFTER_OPENING = True
    OPENING_KERNEL_SIZE = 3
    OPENING_ITERATIONS = 1
    MIN_CONTOUR_AREA = 40
    ASPECT_RATIO_THRESHOLD = 8.0
    SOLIDITY_THRESHOLD = 0.35
    DIFF_THRESHOLD = 30

    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # オートクロップの適用
    if use_auto_crop:
        img1 = auto_rectify(img1)
        img2 = auto_rectify(img2)

    # 位置合わせ（ORBマッチング）
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    try:
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches) * 0.7)]

        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            h_matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
            img_to_compare2 = cv2.warpPerspective(img2, h_matrix, (img1_gray.shape[1], img1_gray.shape[0]))
        else:
            img_to_compare2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    except:
        img_to_compare2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 差分判定ロジック
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_to_compare2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(cv2.GaussianBlur(gray1, (5, 5), 0), cv2.GaussianBlur(gray2, (5, 5), 0))
    _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 彩度差分の考慮
    diff_c = cv2.absdiff(cv2.GaussianBlur(img1, (5, 5), 0), cv2.GaussianBlur(img_to_compare2, (5, 5), 0))
    diff_c_sum = np.sum(diff_c, axis=2).astype('uint8')
    _, thresh_c = cv2.threshold(diff_c_sum, DIFF_THRESHOLD + 20, 255, cv2.THRESH_BINARY)
    
    mask = cv2.bitwise_or(thresh, thresh_c)
    kernel = np.ones((OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPENING_ITERATIONS)
    if DILATE_AFTER_OPENING:
        mask = cv2.dilate(mask, None, iterations=1)

    # 輪郭抽出と枠の描画
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img1.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA: continue
        x, y, wb, hb = cv2.boundingRect(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0 or not (1/ASPECT_RATIO_THRESHOLD < (wb/hb) < ASPECT_RATIO_THRESHOLD): continue
        if (area / hull_area) < SOLIDITY_THRESHOLD: continue
        cv2.rectangle(result_img, (x, y), (x + wb, y + hb), (0, 0, 255), 3)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    result_filename = f'result_{timestamp}.png'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), result_img)
    return result_filename

@app.route('/', methods=['GET'])
def intro():
    """紹介ページを表示"""
    return render_template('intro.html')

@app.route('/main', methods=['GET'])
def index():
    """メイン判定ページを表示"""
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    """比較実行リクエストの処理"""
    data = request.get_json()
    use_auto_crop = data.get('use_auto_crop', False)
    p1 = save_b64_image(data.get('image1_b64'), app.config['UPLOAD_FOLDER'])
    p2 = save_b64_image(data.get('image2_b64'), app.config['UPLOAD_FOLDER'])
    res = compare_images(p1, p2, use_auto_crop=use_auto_crop)
    for p in [p1, p2]: 
        if os.path.exists(p): os.remove(p)
    return jsonify({'result_image': f"/static/uploads/{res}"})

def save_b64_image(b64, folder):
    """Base64形式の画像を保存"""
    data = base64.b64decode(b64.split(",")[1])
    path = os.path.join(folder, f"{uuid.uuid4()}.png")
    with open(path, "wb") as f: f.write(data)
    return path

if __name__ == '__main__':
    # ポート設定を環境変数から取得（デプロイ用）
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)