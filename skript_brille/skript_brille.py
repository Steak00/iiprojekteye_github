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

    # MQTT Setup
    # define broker connection parameters
    mqtt_broker = "localhost"
    mqtt_port = 1883
    mqtt_topic = "eye_tracking/detected_object"

    # create and connect MQTT-Client for communication with website
    mqtt_client = mqtt.Client()
    mqtt_client.connect(mqtt_broker, mqtt_port)

    # send a test message to verify MQTT connection
    mqtt_client.publish(mqtt_topic, "Connection works!")

    # load YOLO model from local path
    model = YOLO("C:/Users/lenaw/iiprojekteyetracking/objectdetection_nano.pt")
    # print model info
    print("Model device:", model.device)
    print(model.names)

    # initialize ZeroMQ Context for communication
    ctx = zmq.Context()

    # set ip and port for Pupil Capture API communication
    ip = 'localhost'
    port = 50020

    # request socket for sending commands to Pupil Capture    
    req_socket = ctx.socket(zmq.REQ)
    req_socket.connect(f'tcp://{ip}:{port}')

    # request the subscription port to receive streaming data
    req_socket.send_string('SUB_PORT')
    sub_port = req_socket.recv_string() # receive actual subscribe port

    # activate gaze plugin which is needed to receive gaze data
    req_socket.send_string('start_plugin gaze')
    req_socket.recv_string()

    # activate the gaze_streaming plugin which streams gaze data via ZMQ
    req_socket.send_string('start_plugin gaze_streaming')
    print("Gaze Stream aktiviert:", req_socket.recv_string())

    # create a subscriber socket to receive gaze data from Pupil Capture
    gaze_socket = ctx.socket(zmq.SUB)
    gaze_socket.connect(f'tcp://{ip}:{sub_port}')
    gaze_socket.setsockopt_string(zmq.SUBSCRIBE, 'gaze')

    # create a subscriber socket to receive the world camera frames
    frame_socket = ctx.socket(zmq.SUB)
    frame_socket.connect(f'tcp://{ip}:{sub_port}')
    frame_socket.setsockopt_string(zmq.SUBSCRIBE, 'frame.world')

    # shared variable for the latest gaze data, protected by a lock for thread-safe access
    latest_gaze = None
    gaze_lock = threading.Lock()

    # shared variable for the latest video frame, protected by a lock
    latest_frame = None
    frame_lock = threading.Lock()

    # shared variable for the latest detected bounding boxes, protected by a lock
    latest_boxes = []
    boxes_lock = threading.Lock()

    # stores the last object class sent over MQTT to prevent duplicate messages and reduce traffic
    last_sent_cls = None

    def gaze_listener():
        # continuously receives gaze data from the gaze stream
        nonlocal latest_gaze
        while True:
            try:
                topic, payload = gaze_socket.recv_multipart()
                gaze_data = msgpack.loads(payload, raw=False)
                norm_pos = gaze_data.get('norm_pos')  # normalized (x,y) gaze position
                if norm_pos:
                    # store te current gaze position thread-safe
                    with gaze_lock:
                        latest_gaze = norm_pos
            except Exception as e:
                print("Fehler beim Empfangen von Gaze:", e)

    def frame_listener():
        # continuously receives frames from the world camera stream
        nonlocal latest_frame
        while True:
            try:
                parts = frame_socket.recv_multipart()
                if len(parts) != 3:
                    continue    # ignores incomplete messages
                topic, msgpack_payload, jpeg_buffer = parts
                img_data = np.frombuffer(jpeg_buffer, dtype=np.uint8)
                frame = cv2.imdecode(img_data, 1)

                # check if frame is valid before using it
                if frame is None or frame.size == 0:
                    print("Warnung: Ungültiger Frame empfangen – übersprungen.")
                    continue

                # store the current frame thread-safe
                if frame is not None:
                    with frame_lock:
                        latest_frame = frame
            except Exception as e:
                print("Fehler beim Empfangen von Frame:", e)

    def yolo_detector():
        # continuously performs object detection on the most recent frame
        nonlocal latest_boxes
        while True:
            frame = None
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()

            if frame is not None:
                try:
                    results = model(frame, imgsz=320)    # run trained YOLO-model
                    boxes = []
                    for result in results:
                        for obj in result.boxes:
                            x1, y1, x2, y2 = map(int, obj.xyxy[0]) # bounding box coordinates
                            cls = model.names[int(obj.cls[0])]    # class name
                            boxes.append((x1, y1, x2, y2, cls))

                    # store the detected object thread-safe
                    with boxes_lock:
                        latest_boxes = boxes
                except Exception as e:
                    print("Fehler bei YOLO-Erkennung:", e)

            time.sleep(0.05) # small delay to reduce CPU usage

    threading.Thread(target=gaze_listener, daemon=True).start()
    threading.Thread(target=frame_listener, daemon=True).start()
    threading.Thread(target=yolo_detector, daemon=True).start()

    print("Threads gestartet. Warte auf Daten...")

    while True:
        try:
            # access the most recent gaze position
            with gaze_lock:
                gaze = latest_gaze

            # access the most recent detected objects
            with boxes_lock:
                boxes = latest_boxes.copy()

            # access the most recent frame
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            # proceed only if both gaze point and frame are available
            if frame is not None and gaze:
                h, w = frame.shape[:2] # get frame dimensions
                # convert normalized gaze coordinates to pixel coordinates
                gaze_x = int(gaze[0] * w) 
                gaze_y = int((1 - gaze[1]) * h)
                gaze_point = (gaze_x, gaze_y)

                highlighted_object = None

                # check if the gaze point is inside any detected bounding box
                for (x1, y1, x2, y2, cls) in boxes:
                    if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                        highlighted_object = (x1, y1, x2, y2, cls)
                        break

                # if a new object is gazed at, publish its class via MQTT
                if highlighted_object:
                    x1, y1, x2, y2, cls = highlighted_object
                    if cls != last_sent_cls:
                        mqtt_client.publish(mqtt_topic, f'"{cls}"') # send class name as JSON string
                        last_sent_cls = cls # update last sent class

                # log information about gaze and detected object
                print(f"Gaze Punkt: ({gaze_x}, {gaze_y}), Bildgröße: ({w}, {h}), Gefundenes Objekt: {highlighted_object[4] if highlighted_object else 'Keines'}")
            else:
                # gaze or frame not yet available
                print("Warte auf Frame oder Gaze Daten...")

            time.sleep(0.01) # small delay to reduce CPU usage

        except Exception as e:
            print("Fehler im Hauptloop:", e)
