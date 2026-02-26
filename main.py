import cv2
from detection.detector import Detector
import os 

os.environ["QT_QPA_PLATFORM"] = "xcb"

def main():
    cap = cv2.VideoCapture(0)
    detector = Detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)

        annotated = results.plot()

        cv2.imshow("Suspicious Behavior Detector", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()