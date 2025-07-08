import cv2
import time
import logging
import serial
import numpy as np
from ultralytics import YOLO
import pymysql
from flask import Flask, render_template, jsonify, request
import threading
from datetime import datetime, timedelta


# 初始化 YOLO 模型
yolo_model = YOLO('yolov8l.pt')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# 等待區的區域座標（畫面中設定的矩形區域）
WAITING_ZONE = [(50, 360), (600, 420)]
drawing = False  # 用來追蹤是否正在繪製矩形
start_point = (0, 0)

# 紅綠燈狀態模擬
RED_LIGHT_TIME = 3  # 紅燈時間（秒）
DETECTION_TIME = 3   # 行人進入等待區後轉綠燈的時間（秒）

# 初始化攝影機
cap = cv2.VideoCapture(0)

# 初始化 Arduino 連接
arduino = serial.Serial('COM3', 115200, timeout=1)  # 根據實際連接的 COM 埠號調整

# 資料庫連線設定
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='',
        db='traffic_system',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# 初始化資料庫
def initialize_database():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS traffic_system;")
            cursor.execute("USE traffic_system;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traffic_light_status (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    light_status ENUM('RED', 'GREEN') NOT NULL
                );
            """)
        connection.commit()
    finally:
        connection.close()

initialize_database()

# Flask 初始化
app = Flask(__name__)

#旋轉監測畫面
def rotate_frame(frame, angle):
    """將影像以中心為軸旋轉指定角度"""
    (h, w) = frame.shape[:2]  # 獲取影像高度與寬度
    center = (w // 2, h // 2)  # 計算影像中心
    # 獲取旋轉矩陣 (M)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # 計算旋轉後的邊界大小，避免影像被裁切
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # 調整旋轉矩陣以考慮新的影像大小
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    # 執行旋轉
    rotated = cv2.warpAffine(frame, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

def send_light_state_to_arduino(state):
    """當燈號改變時，將狀態字符 ('g' 或 'r') 傳回 Arduino"""
    if state == "GREEN":
        arduino.write(b'g')
    elif state == "RED":
        arduino.write(b'r')

def detect_objects(frame, model, classes):
    """使用 YOLO 模型偵測特定類別物件。"""
    results = model.predict(frame)
    detections = results[0].boxes if results else None
    objects = []
    if detections is not None:
        for box in detections:
            cls = int(box.cls[0])
            if cls in classes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                objects.append((cls, x_min, y_min, x_max, y_max, conf))
    return objects

def draw_bounding_boxes(frame, objects, color_map):
    """在影像上繪製邊框，並正確顯示類別標籤。"""
    for obj in objects:
        cls, x_min, y_min, x_max, y_max, conf = obj
        color = color_map.get(cls, (255, 255, 255))  # 根據類別選擇顏色
        # 更新標籤以正確反映類別
        label = f"{'person' if cls == 0 else 'car' if cls == 2 else 'motorcycle' if cls == 3 else 'unknown'} {conf:.2f}"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)  # 繪製邊框
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 繪製標籤

def adjust_waiting_zone(event, x, y, flags, param):
    """滑鼠拉取等待區"""
    global WAITING_ZONE, drawing, start_point

    if event == cv2.EVENT_LBUTTONDOWN:  # 按下滑鼠左鍵
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:  # 滑鼠移動
        if drawing:
            WAITING_ZONE = [start_point, (x, y)]

    elif event == cv2.EVENT_LBUTTONUP:  # 釋放滑鼠左鍵
        drawing = False

        # 確保矩形座標不翻轉
        x1, y1 = start_point
        x2, y2 = x, y
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 限制範圍在畫布內
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(640, x2)
        y2 = min(480, y2)

        WAITING_ZONE = [(x1, y1), (x2, y2)]

def check_pedestrian_in_zone(objects, zone):
    """檢查行人是否在等待區內，排除騎機車的行人，並根據距離條件判斷。"""

    motorcycles = [(x_min, y_min, x_max, y_max) for cls, x_min, y_min, x_max, y_max, conf in objects if cls == 3]  # 機車類別為 3
    for obj in objects:
        cls, x_min, y_min, x_max, y_max, conf = obj
        if cls == 0:  # 類別 0 表示行人
            x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
            if zone[0][0] < x_center < zone[1][0] and zone[0][1] < y_center < zone[1][1]:
                # 判斷行人是否靠近機車
                for mx_min, my_min, mx_max, my_max in motorcycles:
                    if (x_min < mx_max and x_max > mx_min) and (y_min < my_max and y_max > my_min):
                        # 如果行人邊界框與機車邊界框重疊，則忽略該行人
                        return False
                return True
    return False

def traffic_light_control(state, pedestrian_detected, detection_start_time, pedestrian_in_zone, start_time):
    """模擬紅綠燈控制邏輯。"""
    if state == "RED":
        if pedestrian_detected:
            if detection_start_time is None:
                detection_start_time = time.time()
            elif time.time() - detection_start_time >= DETECTION_TIME:
                state = "GREEN"
                pedestrian_in_zone = True
                detection_start_time = None
        else:
            detection_start_time = None
    elif state == "GREEN":
        if not pedestrian_detected:
            if pedestrian_in_zone:
                start_time = time.time()
                pedestrian_in_zone = False
        else:
            start_time = time.time()
            pedestrian_in_zone = True

        if time.time() - start_time >= RED_LIGHT_TIME and not pedestrian_detected:
            state = "RED"           

    return state, detection_start_time, pedestrian_in_zone, start_time

def process_frame(state, start_time, detection_start_time, pedestrian_in_zone, color_map):
    cv2.namedWindow("Pedestrian and Vehicle Detection")
    cv2.setMouseCallback("Pedestrian and Vehicle Detection", adjust_waiting_zone)

    rotation_angle = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # 減少影像處理頻率
        rotated_frame = rotate_frame(frame, rotation_angle)
        objects = detect_objects(rotated_frame, yolo_model, classes=[0, 2, 3])
        draw_bounding_boxes(rotated_frame, objects, color_map)

        # 從 Arduino 獲取sensor狀態

        pedestrian_detected = check_pedestrian_in_zone(objects, WAITING_ZONE)
        prev_state = state
        state, detection_start_time, pedestrian_in_zone, start_time = traffic_light_control(
            state, pedestrian_detected, detection_start_time, pedestrian_in_zone, start_time
        )

        # 當燈號改變時，傳送狀態到 Arduino
        if state != prev_state:
            send_light_state_to_arduino(state)

        # 繪製等待區
        cv2.rectangle(rotated_frame, WAITING_ZONE[0], WAITING_ZONE[1], (0, 0, 255) if state == "RED" else (0, 255, 0), 2)

        # 顯示紅綠燈狀態
        time_left = int(DETECTION_TIME - (time.time() - detection_start_time)) if state == "RED" and detection_start_time else 0
        if state == "RED":
            cv2.putText(rotated_frame, f"Light: RED", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if detection_start_time:
                cv2.putText(rotated_frame, f"Switching in: {time_left}s", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif state == "GREEN":
            cv2.putText(rotated_frame, f"Light: GREEN", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not pedestrian_detected:
                time_left = int(RED_LIGHT_TIME - (time.time() - start_time))
                cv2.putText(rotated_frame, f"Switching to RED in: {time_left}s", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 顯示畫面
        cv2.imshow("Pedestrian and Vehicle Detection", rotated_frame)
        save_light_status(state, prev_state)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):  # 按 'q' 離開
            break
        elif key == ord('a'):  # 'a'向左轉
            rotation_angle -= 5
        elif key == ord('d'):  # 'd'向右轉
            rotation_angle += 5

    cap.release()
    cv2.destroyAllWindows()

def main():
    state = "RED"
    start_time = time.time()
    detection_start_time = None
    pedestrian_in_zone = False

    color_map = {
        0: (189, 192, 186),  # 行人：白鼠
        2: (255, 0, 0),  # 汽車：藍色
        3: (0, 0, 255)   # 機車：紅色
    }

    process_frame(state, start_time, detection_start_time, pedestrian_in_zone, color_map)

    """處理影像幀，進行物件偵測和交通燈控制"""
    cv2.namedWindow("Pedestrian and Vehicle Detection")
    cv2.setMouseCallback("Pedestrian and Vehicle Detection", adjust_waiting_zone)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 將畫面旋轉 5 度
        rotated_frame = rotate_frame(frame, 5)

        objects = detect_objects(rotated_frame, yolo_model, classes=[0, 2, 3])
        draw_bounding_boxes(rotated_frame, objects, color_map)

        pedestrian_detected = check_pedestrian_in_zone(objects, WAITING_ZONE)
        state, detection_start_time, pedestrian_in_zone, start_time = traffic_light_control(
            state, pedestrian_detected, detection_start_time, pedestrian_in_zone, start_time
        )

        # 繪製等待區
        cv2.rectangle(rotated_frame, WAITING_ZONE[0], WAITING_ZONE[1], (0, 0, 255) if state == "RED" else (0, 255, 0), 2)

        # 顯示紅綠燈狀態
        time_left = int(DETECTION_TIME - (time.time() - detection_start_time)) if state == "RED" and detection_start_time else 0
        if state == "RED":
            cv2.putText(rotated_frame, f"Light: RED", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if detection_start_time:
                cv2.putText(rotated_frame, f"Switching in: {time_left}s", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif state == "GREEN":
            cv2.putText(rotated_frame, f"Light: GREEN", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not pedestrian_detected:
                time_left = int(RED_LIGHT_TIME - (time.time() - start_time))
                cv2.putText(rotated_frame, f"Switching to RED in: {time_left}s", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 顯示畫面
        cv2.imshow("Pedestrian and Vehicle Detection", rotated_frame)

        # 儲存燈號狀態到資料庫
        save_light_status(state, prev_state)

        # 更新前一個燈號狀態
        prev_state = state

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_light_status(state, prev_state):
    """僅當燈號變化時才寫入資料庫"""
    if state != prev_state:
        try:
            conn = pymysql.connect(
                host='localhost',
                port=3306,
                user='root',
                password='',  # 若有設定密碼，請填寫
                db='traffic_system',
                charset='utf8mb4'
            )
            with conn.cursor() as cursor:
                query = "INSERT INTO traffic_light_status (timestamp, light_status) VALUES (%s, %s)"
                # 使用當前時間
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute(query, (timestamp, state))
                conn.commit()
                print("✅ 燈號狀態已儲存到資料庫")
        except pymysql.MySQLError as e:
            print(f"❌ 資料庫連線失敗：{e}")
        finally:
            conn.close()

# 資料庫連接設定
def get_traffic_light_status():
    try:
        conn = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='',  # 填寫您的 MySQL 密碼
            db='traffic_system',
            charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            # 查詢最新的10條紀錄
            query = "SELECT timestamp, light_status FROM traffic_light_status ORDER BY timestamp DESC LIMIT 10"
            cursor.execute(query)
            result = cursor.fetchall()
            return result
    except pymysql.MySQLError as e:
        print(f"資料庫錯誤: {e}")
        return []
    finally:
        conn.close()

@app.route('/')
def index():
    records = get_traffic_light_status()  # 獲取資料庫中的資料
    return render_template('index.html', records=records)

@app.route("/chart")
def chart():
    return render_template("chart.html")

@app.route("/data")
def get_data():
    start = request.args.get('start')
    end = request.args.get('end')

    # 如果未提供 start 和 end 參數，則使用最近七天的範圍
    if not start or not end:
        end = datetime.today().strftime('%Y-%m-%d')
        start = (datetime.today() - timedelta(days=6)).strftime('%Y-%m-%d')

    query = "SELECT date, green_light_count FROM daily_green_light_count WHERE date BETWEEN %s AND %s ORDER BY date ASC"
    params = [start, end]

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        data = cursor.fetchall()
    conn.close()

    return jsonify(data)


# 啟動 Flask 伺服器
def start_flask():
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    # 開啟 Flask 伺服器
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.start()

    # 進行影像偵測和交通燈控制
    main()