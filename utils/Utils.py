import cv2, os
import traceback

class Utils:


    def __init__(self):
        print('Utils init')


    #  openCv로 비디오를 읽어 프레임으로 짜릅니다.
    def extract_frames(self,video_path, output_dir, every_n_frames=5):
        try:
            os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            count = 0
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if i % every_n_frames == 0:
                    cv2.imwrite(f"{output_dir}/frame_{count:04d}.jpg", frame)
                    count += 1
                i += 1
            cap.release()
        except:
            traceback.print_exc()