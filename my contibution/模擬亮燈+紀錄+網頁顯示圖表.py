import cv2
import time
import numpy as np
from ultralytics import YOLO
import pymysql
from flask import Flask, render_template, url_for, jsonify, request
import threading
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import mysql.connector

# 初始化 YOLO 模型
yolo_model = YOLO('yolov8l.pt')


# 等待區的區域座標
WAITING_ZONE = [(50, 360), (600, 420)]
RED_LIGHT_TIME = 7
DETECTION_TIME = 3

# 初始化 Flask 應用
app = Flask(__name__)

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

# 前端首頁
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



# 實時處理影像幀
def process_frame():
    state = "RED"
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 模擬切換紅綠燈狀態
        current_time = time.time()
        if state == "RED" and current_time % 10 < 5:
            state = "GREEN"
            switch_time = datetime.now()
            save_light_status_to_db(state, switch_time)
        elif state == "GREEN" and current_time % 10 >= 5:
            state = "RED"
            switch_time = datetime.now()
            save_light_status_to_db(state, switch_time)

        # 顯示紅綠燈狀態
        color = (0, 255, 0) if state == "GREEN" else (0, 0, 255)
        cv2.rectangle(frame, WAITING_ZONE[0], WAITING_ZONE[1], color, 2)
        cv2.putText(frame, f"Light: {state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Traffic Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 儲存紅綠燈狀態至資料庫
def save_light_status_to_db(state, switch_time):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO traffic_light_status (timestamp, light_status)
                VALUES (%s, %s);
            """, (switch_time, state))
        connection.commit()
    finally:
        connection.close()

if __name__ == '__main__':
    
    flask_thread = threading.Thread(target=app.run, kwargs={'debug': True, 'use_reloader': False})
    flask_thread.start()
    

    process_frame()
