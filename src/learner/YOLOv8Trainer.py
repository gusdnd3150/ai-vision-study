
from ultralytics import YOLO
import cv2

class YOLOv8Trainer:

    model = None

    def __init__(self,model_name):
        print('YOLOv8Trainer init')
        self.model_name = model_name


    def create_model(self):
        print('create_model')
        self.model = YOLO("yolov8m.pt")

    def load_model(self):
        print('load_model')
        model = YOLO("yolov8m.pt")


    def learn_model(self):
        print('learn_model')


    def predict_image(self, image_path):
        print('predict_image')

    def run_opencv(self):
        # 웹캠 열기
        cap = cv2.VideoCapture(0)  # 0: 기본 카메라

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 추론
            results = self.model.predict(source=frame, conf=0.5, verbose=False)

            # 사람(person)만 필터링
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])  # 클래스 id
                    if cls_id == 0:  # COCO에서 'person' 클래스 ID = 0
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # 결과 화면 출력
            cv2.imshow("Person Detection", frame)

            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()