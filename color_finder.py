from djitellopy import Tello
import cv2 as cv
import datetime, time, threading
import numpy as np

class ColorDetector:
    def __init__(self, should_use_tello: bool, views: bool, logs: bool, tello = -1):
        self.use_tello = should_use_tello
        self.split_views = views
        self.use_debug_log = logs
        self.tello_inst = tello

    def _find_color(self, color_name: str, lower_bound: int, upper_bound: int, frame):
        lower = np.array(lower_bound)
        upper = np.array(upper_bound)
        mask = cv.inRange(frame, lower, upper)

        if self.use_tello:
            lower[0], lower[2] = lower[2], lower[0]
            upper[0], upper[2] = upper[2], upper[0]

        has_color = np.sum(mask)
        if has_color > 0 and self.use_debug_log:
            print(f'{color_name} Spotted at {datetime.datetime.now()}')

        contours, _ = cv.findContours(mask,
                                            cv.RETR_EXTERNAL,
                                            cv.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv.boundingRect(contour)
                frame = cv.rectangle(frame, (x, y),
                                        (x + w, y + h),
                                        (255, 255, 255), 2)

                cv.putText(frame, color_name, (x, y-32),
                            cv.FONT_HERSHEY_DUPLEX, 1.0,
                            (0, 0, 0), 9)

                cv.putText(frame, color_name, (x, y-32),
                            cv.FONT_HERSHEY_DUPLEX, 1.0,
                            (255, 255, 255))

        result = cv.bitwise_and(frame, frame, mask=mask)
        return result

    def detect_colors(self):
        title = "HorsePower Tello Monitor"

        # setup and connect to tello
        if self.use_tello:
            if self.tello_inst == -1:
                print("Create Tello object")
                tello = Tello()

                print("Connect to Tello Drone")
                tello.connect()

                battery_level = tello.get_battery()
                print(f"Battery Life Percentage: {battery_level}")
            else:
                tello = self.tello_inst

            print("Turn Video Stream On")
            tello.streamon()

            print("Read Tello Image")
            frame_read = tello.get_frame_read()
        else:
            cap = cv.VideoCapture(0)

        while True:
            if self.use_tello:
                frame = frame_read.frame
            else:
                _, frame = cap.read()

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            #frame = cv.medianBlur(frame, 5)

            red = self._find_color("Red", [0,0,50], [100,33,240], frame)
            blue = self._find_color("Blue", [100,20,0], [255,106,65], frame)

            combined_mask = red + blue

            color_correct = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            final_frame_color = frame if self.use_tello == True else color_correct

            final = final_frame_color if self.split_views == False else np.hstack((final_frame_color, combined_mask))

            cv.imshow(title, final)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        if self.use_tello:
            tello.streamoff()
        else:
            cap.release()

        cv.destroyWindow(title)
        cv.destroyAllWindows()

    def run(self):
        thread = threading.Thread(target=self.detect_colors)
        thread.start()

def main():
    detector = ColorDetector(False, False, False)
    detector.detect_colors()

if __name__ == "__main__":
    main()