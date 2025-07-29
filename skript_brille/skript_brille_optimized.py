# skript_brille.py

def start():
    import zmq
    import msgpack
    import cv2
    import numpy as np
    import threading
    from ultralytics import YOLO
    import paho.mqtt.client as mqtt
    import time
    import subprocess

    print('success')
    #subprocess.Popen(["C:/Program Files (x86)/Pupil-Labs/Pupil v3.5.1/Pupil Capture v3.5.1/pupil_capture.exe"])
    #time.sleep(5)

    mqtt_broker = "localhost"
    mqtt_port = 1883
    mqtt_topic = "eye_tracking/detected_object"

    mqtt_client = mqtt.Client()
    mqtt_client.connect(mqtt_broker, mqtt_port)
    mqtt_client.publish(mqtt_topic, "Connection works!")

    model = YOLO("C:/Users/lenaw/iiprojekteyetracking/objectdetection_nano.pt")
    print("Model device:", model.device)
    print(model.names)

    ctx = zmq.Context()
    ip = 'localhost'
    port = 50020

    req_socket = ctx.socket(zmq.REQ)
    req_socket.connect(f'tcp://{ip}:{port}')
    req_socket.send_string('SUB_PORT')
    sub_port = req_socket.recv_string()

   # req_socket.send_string('R pupil.0')
    #req_socket.recv_string()

   # req_socket.send_string('R pupil.1')
    #req_socket.recv_string()
    #req_socket.send_string('start_plugin pupil')
    #print("Start pupil plugin:", req_socket.recv_string())

    req_socket.send_string('start_plugin gaze')
    req_socket.recv_string()

    req_socket.send_string('start_plugin gaze_streaming')
    print("Gaze Stream aktiviert:", req_socket.recv_string())

    gaze_socket = ctx.socket(zmq.SUB)
    gaze_socket.connect(f'tcp://{ip}:{sub_port}')
    gaze_socket.setsockopt_string(zmq.SUBSCRIBE, 'gaze')

    frame_socket = ctx.socket(zmq.SUB)
    frame_socket.connect(f'tcp://{ip}:{sub_port}')
    frame_socket.setsockopt_string(zmq.SUBSCRIBE, 'frame.world')

    latest_gaze = None
    gaze_lock = threading.Lock()

    latest_frame = None
    frame_lock = threading.Lock()

    latest_boxes = []
    boxes_lock = threading.Lock()

    last_sent_cls = None

    def gaze_listener():
        nonlocal latest_gaze
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

    def frame_listener():
        nonlocal latest_frame
        while True:
            try:
                parts = frame_socket.recv_multipart()
                if len(parts) != 3:
                    continue
                topic, msgpack_payload, jpeg_buffer = parts
                img_data = np.frombuffer(jpeg_buffer, dtype=np.uint8)
                frame = cv2.imdecode(img_data, 1)
                if frame is None or frame.size == 0:
                    print("Warnung: Ungültiger Frame empfangen – übersprungen.")
                    continue

                if frame is not None:
                    with frame_lock:
                        latest_frame = frame
            except Exception as e:
                print("Fehler beim Empfangen von Frame:", e)

    def yolo_detector():
        nonlocal latest_boxes
        while True:
            frame = None
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()

            if frame is not None:
                try:
                    results = model(frame, imgsz=320)
                    boxes = []
                    for result in results:
                        for obj in result.boxes:
                            x1, y1, x2, y2 = map(int, obj.xyxy[0])
                            cls = model.names[int(obj.cls[0])]
                            boxes.append((x1, y1, x2, y2, cls))

                    with boxes_lock:
                        latest_boxes = boxes
                except Exception as e:
                    print("Fehler bei YOLO-Erkennung:", e)

            time.sleep(0.05)

    threading.Thread(target=gaze_listener, daemon=True).start()
    threading.Thread(target=frame_listener, daemon=True).start()
    threading.Thread(target=yolo_detector, daemon=True).start()

    print("Threads gestartet. Warte auf Daten...")

    while True:
        try:
            with gaze_lock:
                gaze = latest_gaze

            with boxes_lock:
                boxes = latest_boxes.copy()

            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            if frame is not None and gaze:
                h, w = frame.shape[:2]
                gaze_x = int(gaze[0] * w)
                gaze_y = int((1 - gaze[1]) * h)
                gaze_point = (gaze_x, gaze_y)

                # Nur den Bereich um den gaze-Punkt an das Objekterkennungsmodell weitergeben
                crop_size = 300  # kleinerer Wert = weniger Rechenaufwand, aber evtl. weniger Objekterkennung
                crop_x1 = max(gaze_x - crop_size, 0)
                crop_y1 = max(gaze_y - crop_size, 0)
                crop_w = min(gaze_x + crop_size, w) - crop_x1
                crop_h = min(gaze_y + crop_size, h) - crop_y1

                frame_crop = frame[crop_y1:crop_y1+crop_h, crop_x1:crop_x1+crop_w]
                
                highlighted_object = None
                try:
                    results = model(frame_crop)
                    for result in results:
                        for obj in result.boxes:
                            x1, y1, x2, y2 = map(int, obj.xyxy[0])
                            # Koordinaten auf das Originalbild verschieben
                            x1 += crop_x1
                            y1 += crop_y1
                            x2 += crop_x1
                            y2 += crop_y1
                            cls = model.names[int(obj.cls[0])]
                            if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                                highlighted_object = (x1, y1, x2, y2, cls)
                                break
                except Exception as e:
                    print("Fehler bei YOLO-Erkennung im Crop:", e)
                
                if highlighted_object:
                    x1, y1, x2, y2, cls = highlighted_object
                    if cls != last_sent_cls:
                        mqtt_client.publish(mqtt_topic, f'"{cls}"')
                        last_sent_cls = cls


                print(f"Gaze Punkt: ({gaze_x}, {gaze_y}), Bildgröße: ({w}, {h}), Gefundenes Objekt: {highlighted_object[4] if highlighted_object else 'Keines'}")
            else:
                print("Warte auf Frame oder Gaze Daten...")

            time.sleep(0.01)

        except Exception as e:
            print("Fehler im Hauptloop:", e)

# NICHT automatisch starten
# if __name__ == "__main__":
#     start()
