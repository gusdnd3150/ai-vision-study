from src.learner.StaticClassifierTrainer import StaticClassifierTrainer
import cv2

from src.learner.YOLOv8Trainer import YOLOv8Trainer


class Init:

    imgTrain = None

    def __init__(self):
        print('Init init')

        # self.runVedioModel()

        test = YOLOv8Trainer('test')
        test.create_model()
        test.run_opencv()



    def runStaticModel(self):
        print('runStaticModel')
        self.imgTrain = StaticClassifierTrainer(
            'ball_model'
            ,'models/'
            ,'resources/trainData/static'
            ,'resources/eval/static'
            ,2
        )
        # self.imgTrain.run_train_process()
        self.imgTrain.loadModel()

        resutl = self.imgTrain.predict_image('resources/test/test.jpg')
        print(f'result : {resutl}')




    def runVedioModel(self):
        print('runVedioModel')

        # model = load_model()
        cap = cv2.VideoCapture(0)  # or 0 for webcam
        if not cap.isOpened():
            print("비디오를 열 수 없습니다:", 0)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # inp = preprocess(frame)
            # 모델은 리스트 형태의 배치 입력을 받음
            # outputs = model([inp])
            # out0 = outputs[0]
            # boxes = out0['boxes'].cpu().numpy()
            # labels = out0['labels'].cpu().numpy()
            # scores = out0['scores'].cpu().numpy()
            # vis = draw_boxes(frame, boxes, labels, scores)
            # out.write(vis)
            # 화면에 실시간 표시(원하면)
            # cv2.imshow("result", vis)
            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        #
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))
        #
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # print("DEVICE:", DEVICE, "FPS:", fps, "SIZE:", w, h, "FRAMES:", frame_count)
        #
        # with torch.no_grad():
        #     while True:
        #         ret, frame = cap.read()
        #         if not ret:
        #             break
        #         inp = preprocess(frame)
        #         # 모델은 리스트 형태의 배치 입력을 받음
        #         outputs = model([inp])
        #         out0 = outputs[0]
        #         boxes = out0['boxes'].cpu().numpy()
        #         labels = out0['labels'].cpu().numpy()
        #         scores = out0['scores'].cpu().numpy()
        #         vis = draw_boxes(frame, boxes, labels, scores)
        #         out.write(vis)
        #         # 화면에 실시간 표시(원하면)
        #         cv2.imshow("result", vis)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #
        # cap.release()
        # out.release()
        # cv2.destroyAllWindows()
        # print("완료. 출력:", VIDEO_OUT)