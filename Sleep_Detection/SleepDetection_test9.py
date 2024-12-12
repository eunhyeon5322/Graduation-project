import cv2
import numpy as np
import dlib
import math
import time
from datetime import datetime
import mysql.connector
from mysql.connector import Error

# MySQL 연결 설정
try:
    connection = mysql.connector.connect(host='localhost',
                                         database='driverdata',
                                         user='root',
                                         password='root')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
except Error as e:
    print("Error while connecting to MySQL", e)

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

frame_width = 640
frame_height = 480

title_name = 'Drowsiness Detection'

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_name)

predictor_path = "C:/Sleep_Detection/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

class KalmanFilter:
    def __init__(self, process_variance, estimated_measurement_variance, initial_estimate):
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = initial_estimate
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

# 고개 회전 감지에 사용할 칼만 필터
head_pose_filter = KalmanFilter(process_variance=1e-5, estimated_measurement_variance=0.1, initial_estimate=0.0)

def get_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 카메라 장치에 접근
cap = cv2.VideoCapture(0)

# 카메라 장치에 접근할 수 있는지 확인
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ear_values = []

process_variance = 1e-5
estimated_measurement_variance = 0.1
initial_estimate = 0.5

eye_ear_filter = KalmanFilter(process_variance, estimated_measurement_variance, initial_estimate)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)

    mask = np.zeros_like(frame_gray)
    mask[:] = 255

    face_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
        face_detected = True

    blurred = cv2.blur(frame, (30, 30))
    frame = np.where(mask[..., None] == 255, blurred, frame)

    if face_detected:
        (x, y, w, h) = faces[0]
        frame_with_face = np.copy(frame)

        cv2.rectangle(frame_with_face, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = frame_gray[y:y + h, x:x + w]
        roi_color = frame_with_face[y:y + h, x:x + w]
        landmarks = predictor(roi_gray, dlib.rectangle(0, 0, w, h))

        left_eye = np.array([(landmarks.part(n).x + x, landmarks.part(n).y + y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x + x, landmarks.part(n).y + y) for n in range(42, 48)])

        left_ear = get_ear(left_eye)
        right_ear = get_ear(right_eye)
        mean_ear = (left_ear + right_ear) / 2

        filtered_ear = eye_ear_filter.update(mean_ear)

        ear_values.append(mean_ear)

        cv2.putText(frame_with_face, f"EAR: {mean_ear:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imshow("Drowsiness Detection", frame_with_face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if ear_values:
    average_ear = sum(ear_values) / len(ear_values)
    print(f"Calculated Average EAR: {average_ear}")

    min_EAR = average_ear - 0.02
    print(f"Assigned min_EAR: {min_EAR}")
else:
    print("No EAR values were calculated.")

status = 'Awake'
number_closed = 0
closed_limit = 10
sign = None
color = None

def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def calculate_head_pose(landmarks):
    image_points = np.array([
        (landmarks[30][0], landmarks[30][1]),  # 코 끝
        (landmarks[8][0], landmarks[8][1]),    # 턱
        (landmarks[36][0], landmarks[36][1]),  # 왼쪽 눈 구석
        (landmarks[45][0], landmarks[45][1]),  # 오른쪽 눈 구석
        (landmarks[48][0], landmarks[48][1]),  # 왼쪽 입 모서리
        (landmarks[54][0], landmarks[54][1])   # 오른쪽 입 모서리
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # 코
        (0.0, -330.0, -65.0),        # 턱
        (-225.0, 170.0, -135.0),     # 왼쪽 눈
        (225.0, 170.0, -135.0),      # 오른쪽 눈
        (-150.0, -150.0, -125.0),    # 왼쪽 입 모서리
        (150.0, -150.0, -125.0)      # 오른쪽 입 모서리
    ])

    focal_length = (640, 640)
    center = (320, 240)
    camera_matrix = np.array(
        [[focal_length[0], 0, center[0]],
         [0, focal_length[1], center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    (_, rotation_vector, _) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, np.zeros((3, 1))))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))

    return pitch, yaw, roll

# 칼만 필터를 사용하여 고개 회전 각도를 감지합니다.
def detect_head_rotation(pitch):
    filtered_pitch = head_pose_filter.update(pitch)
    return filtered_pitch

cap = cv2.VideoCapture(0)
time.sleep(2.0)

# MySQL에 데이터 삽입하는 함수
def insert_data_to_database(user_id, sleep, face_shaking, detection_date, detection_time):
    try:
        if connection.is_connected():
            cursor.execute("INSERT INTO sleepdetection (user_id, sleep, face_shaking, detection_date, detection_time) "
                           "VALUES (%s, %s, %s, %s, %s)",
                           (user_id, sleep, face_shaking, detection_date, detection_time))
            connection.commit()
            #print("Data inserted successfully into MySQL database")
    except mysql.connector.IntegrityError as e:
        if e.errno == 1062:  # UNIQUE constraint violation error code
            print("Duplicate entry. Ignoring...")
        else:
            print("Error inserting data into MySQL:", e)

# 고개 회전 감지 및 처리
def detect_head_rotation(pitch):
    filtered_pitch = head_pose_filter.update(pitch)
    return filtered_pitch

# 이전 고개 회전 상태 저장 변수
prev_head_rotation = None

# 이전 프레임에서의 고개 회전 각도 초기화
prev_yaw = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)

    mask = np.zeros_like(frame_gray)
    mask[:] = 255

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)

        blurred = cv2.blur(frame, (30, 30))
        frame = np.where(mask[..., None] == 255, blurred, frame)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
        show_parts = points[EYES]
        right_eye_EAR = get_ear(points[RIGHT_EYE])
        left_eye_EAR = get_ear(points[LEFT_EYE])
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2

        right_eye_center = np.mean(points[RIGHT_EYE], axis=0).astype("int")
        left_eye_center = np.mean(points[LEFT_EYE], axis=0).astype("int")

        cv2.putText(frame, "{:.2f}".format(right_eye_EAR), (right_eye_center[0, 0], right_eye_center[0, 1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "{:.2f}".format(left_eye_EAR), (left_eye_center[0, 0], left_eye_center[0, 1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for (i, point) in enumerate(show_parts):
            x = point[0, 0]
            y = point[0, 1]
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

        if mean_eye_EAR > min_EAR:
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = 0
        else:
            color = (0, 128, 255)
            number_closed += 1

        if number_closed >= closed_limit:
            color = (0, 0, 255)
            status = 'Sleep'

            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            if number_closed % 10 == 0 and prev_head_rotation != 'sleep':
                num = number_closed // 10
                insert_data_to_database(1, 1, 0, date, time)
                print(f"({num}times)/ Date: {date}, Time: {time}")

        sign = 'sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)

        if number_closed > closed_limit:
            pass

        cv2.putText(frame, status, (x - w, y - h), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)
        cv2.putText(frame, sign, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        shape = predictor(frame_gray, dlib.rectangle(x, y, x + w, y + h))
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        pitch, yaw, _ = calculate_head_pose(landmarks)
        filtered_pitch = detect_head_rotation(pitch)

        if pitch < -35:
            direction = "sleep"
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")
            #print(f"Direction: {direction}, Date: {date}, Time: {time}")
            if prev_head_rotation != 'sleep':
                insert_data_to_database(1, 0, 1, date, time)
            prev_head_rotation = 'sleep'
        else:
            prev_head_rotation = None


    cv2.imshow(title_name, frame)

    if cv2.waitKey(250) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
