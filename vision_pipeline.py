import ast
import queue
import threading
import numpy as np
import cv2

from config import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD,
    SEGMENTATION_INPUT_SIZE, POSE_INPUT_SIZE,
    TARGET_CLASSES, FRAME_SKIP, COLOR_TARGETS,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class YOLOVisionPipeline:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.class_names = {}

        self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                YOLO_MODEL_PATH,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            ## COCO class names stored in the ultralytics onnx metadata
            meta = self.session.get_modelmeta().custom_metadata_map
            self.class_names = ast.literal_eval(meta["names"]) if "names" in meta else {}
            print(f"[YOLOVisionPipeline] Model loaded: {YOLO_MODEL_PATH}")
            print(f"[YOLOVisionPipeline] Input shape: {self.input_shape}")
        except Exception as e:
            print(f"[WARN] YOLO model failed: {e}")
            print(f"[INFO] Using color detection fallback")
            self.session = None

    def preprocess(self, frame, input_size):
        h, w = frame.shape[:2]
        input_w, input_h = input_size
        scale = min(input_w / w, input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = input_w - new_w
        dh = input_h - new_h
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, left, top

    def postprocess_seg(self, outputs, scale, left, top, orig_shape):
        predictions = outputs[0][0]
        protos = outputs[1][0]

        pred_mask = predictions[4:-32] if outputs[0].shape[1] > 4 else predictions[4:]
        pred_mask = None

        boxes = []
        scores = []
        class_ids = []
        mask_coefficients = []

        predictions = np.transpose(predictions)

        for pred in predictions:
            scores_all = pred[4:-32] if outputs[0].shape[1] > 4 else pred[4:]
            class_id = np.argmax(scores_all)
            confidence = scores_all[class_id]

            if confidence < YOLO_CONFIDENCE_THRESHOLD:
                continue

            cx = (pred[0] - left) / scale
            cy = (pred[1] - top) / scale
            w = pred[2] / scale
            h = pred[3] / scale

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_shape[1], x2)
            y2 = min(orig_shape[0], y2)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(class_id)
            mask_coefficients.append(pred[-32:])

        if len(boxes) == 0:
            return [], [], []

        indices = cv2.dnn.NMSBoxes(boxes, scores, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD)

        result_boxes = []
        result_scores = []
        result_class_ids = []

        if len(indices) > 0:
            for i in indices.flatten() if len(indices.shape) > 1 else indices:
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_class_ids.append(class_ids[i])

        return result_boxes, result_scores, result_class_ids

    def postprocess_pose(self, outputs, scale, left, top, orig_shape):
        predictions = outputs[0][0]
        predictions = np.transpose(predictions)

        keypoints_list = []

        for pred in predictions:
            scores_all = pred[4:]
            class_id = 0
            confidence = scores_all[0]

            if confidence < YOLO_CONFIDENCE_THRESHOLD:
                continue

            cx = (pred[0] - left) / scale
            cy = (pred[1] - top) / scale
            w = pred[2] / scale
            h = pred[3] / scale

            kps = pred[4 + 1:].reshape(-1, 3)
            for kp in kps:
                kp[0] = (kp[0] - left) / scale
                kp[1] = (kp[1] - top) / scale

            keypoints_list.append({
                "box": [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)],
                "confidence": float(confidence),
                "keypoints": kps.tolist(),
            })

        return keypoints_list

    def detect_colored_balls(self, frame):
        ## color based detection for test spheres
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_balls = []

        for color_name, color_config in COLOR_TARGETS.items():
            lower = np.array(color_config["lower"])
            upper = np.array(color_config["upper"])

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)

                if color_config["radius_min"] <= radius <= color_config["radius_max"]:
                    area = cv2.contourArea(contour)
                    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2) if cv2.arcLength(contour, True) > 0 else 0

                    if circularity > 0.6:
                        x1 = int(cx - radius)
                        y1 = int(cy - radius)
                        x2 = int(cx + radius)
                        y2 = int(cy + radius)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        class_id = TARGET_CLASSES.index(color_name) if color_name in TARGET_CLASSES else -1
                        if class_id >= 0:
                            detected_balls.append({
                                "box": [x1, y1, x2, y2],
                                "score": circularity,
                                "class_id": class_id,
                                "color": color_name,
                            })

        return detected_balls

    def inference_worker(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                break

            seg_boxes = []
            seg_scores = []
            seg_labels = []

            ## YOLO seg inference if model loaded
            if self.session:
                try:
                    blob, scale, left, top = self.preprocess(frame, SEGMENTATION_INPUT_SIZE)
                    outputs = self.session.run(None, {self.input_name: blob})
                    boxes, scores, class_ids = self.postprocess_seg(outputs, scale, left, top, frame.shape)
                    seg_boxes += boxes
                    seg_scores += scores
                    seg_labels += [self.class_names.get(int(c), f"class_{c}") for c in class_ids]
                except Exception as e:
                    print(f"[WARN] YOLO inference error: {e}")

            ## color detection always runs so balls show up even with a person in frame
            try:
                for ball in self.detect_colored_balls(frame):
                    seg_boxes.append(ball["box"])
                    seg_scores.append(ball["score"])
                    seg_labels.append(ball["color"])
            except Exception as e:
                print(f"[WARN] color detection error: {e}")

            self.result_queue.put({
                "seg_boxes": seg_boxes,
                "seg_scores": seg_scores,
                "seg_labels": seg_labels,
            })

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.thread.start()
        print("[YOLOVisionPipeline] Inference thread started")

    def stop(self):
        self.running = False
        if self.thread:
            self.frame_queue.put(None)
            self.thread.join(timeout=5)
        print("[YOLOVisionPipeline] Stopped")

    def process_frame(self, frame):
        if self.frame_queue.full():
            return None
        self.frame_queue.put(frame.copy())
        try:
            return self.result_queue.get(timeout=2.0)
        except queue.Empty:
            return None

    def process_frame_async(self, frame):
        ## non blocking: queue frame if worker is free, return newest result or None
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
