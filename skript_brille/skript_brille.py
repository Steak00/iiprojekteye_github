import zmq
import msgpack
import cv2
import numpy as np
import threading
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import subprocess

# Pupil capture automatisch starten
subprocess.Popen(["C:/Program Files (x86)/Pupil-Labs/Pupil v3.5.1/Pupil Capture v3.5.1/pupil_capture.exe"])
#MQTT Setup
mqtt_broker = "localhost"
mqtt_port = 1883
mqtt_topic = "eye_tracking/detected_object"

mqtt_client = mqtt.Client()
mqtt_client.connect(mqtt_broker, mqtt_port)
mqtt_client.publish(mqtt_topic, "Connection works!")

# YOLO Modell laden
#model = YOLO("yolov8n.pt")
model = YOLO("C:/Users/lenaw/iiprojekteyetracking/objectdetection_nano.pt")
print(model.names)

# ZMQ Setup
ctx = zmq.Context()
ip = 'localhost'
port = 50020

# 1. Request SUB_PORT
req_socket = ctx.socket(zmq.REQ)
req_socket.connect(f'tcp://{ip}:{port}')
req_socket.send_string('SUB_PORT')
sub_port = req_socket.recv_string()

# Activate plugins
req_socket.send_string('R pupil.0')
req_socket.recv_string()

req_socket.send_string('start_plugin gaze')
req_socket.recv_string()

req_socket.send_string('start_plugin gaze_streaming')
print("Gaze Stream aktiviert:", req_socket.recv_string())

# Setup ZMQ sockets
gaze_socket = ctx.socket(zmq.SUB)
gaze_socket.connect(f'tcp://{ip}:{sub_port}')
gaze_socket.setsockopt_string(zmq.SUBSCRIBE, 'gaze')

frame_socket = ctx.socket(zmq.SUB)
frame_socket.connect(f'tcp://{ip}:{sub_port}')
frame_socket.setsockopt_string(zmq.SUBSCRIBE, 'frame.world')

# Shared gaze data
latest_gaze = None
gaze_lock = threading.Lock()

# Gaze listener thread
def gaze_listener():
    global latest_gaze
    while True:
        try:
            topic, payload = gaze_socket.recv_multipart()
            gaze_data = msgpack.loads(payload, raw=False)
            norm_pos = gaze_data.get('norm_pos')
            if norm_pos:
                with gaze_lock:
                    latest_gaze = norm_pos
        except Exception as e:
            print("Fehler beim Empfangen von Gaze:", e)

threading.Thread(target=gaze_listener, daemon=True).start()

# Main frame loop
while True:
    try:
        parts = frame_socket.recv_multipart()
        if len(parts) != 3:
            continue
        topic, msgpack_payload, jpeg_buffer = parts
        img_data = np.frombuffer(jpeg_buffer, dtype=np.uint8)
        frame = cv2.imdecode(img_data, 1)

        with gaze_lock:
            gaze = latest_gaze

        if frame is not None and gaze:
            h, w = frame.shape[:2]
            gaze_x = int(gaze[0] * w)
            gaze_y = int((1 - gaze[1]) * h)
            gaze_point = (gaze_x, gaze_y)

            # YOLO Objekterkennung
            results = model(frame)
            highlighted_object = None

            for result in results:
                for obj in result.boxes:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    cls = model.names[int(obj.cls[0])]

                    # Prüfen, ob Gaze im Objekt liegt
                    if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                        highlighted_object = (x1, y1, x2, y2, cls)
                        print(f"Gaze befindet sich auf Objekt: {cls}")
                        #break  # Nur das erste passende Objekt markieren

                        mqtt_client.publish(mqtt_topic, f'"{cls}"')
                    break

                if highlighted_object:
                    break

            # Objekt hervorheben
            if highlighted_object:
                x1, y1, x2, y2, cls = highlighted_object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Gaze-Punkt anzeigen
            cv2.circle(frame, gaze_point, 8, (255, 0, 0), -1)
            cv2.putText(frame, f"Gaze: ({gaze[0]:.2f}, {gaze[1]:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Gaze-basierte Objekterkennung", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Fehler beim Verarbeiten des Frames:", e)
        continue

cv2.destroyAllWindows()
