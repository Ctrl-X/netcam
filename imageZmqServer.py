# run this program on the Mac to display image streams from multiple RPis

import time
import cv2
import imagezmq



class FpsCatcher:
    currentMilliTime = lambda: int(round(time.time() * 1000))

    def __init__(self):
        self.initTime()

    def initTime(self):
        self.currentTime = 0
        self.currentFrame = 0
        self.fps = 0

    def compute(self):
        now = FpsCatcher.currentMilliTime()
        if now - self.currentTime >= 1000:
            self.currentTime = now
            self.fps = self.currentFrame
            self.currentFrame = 0
        self.currentFrame += 1



image_hub = imagezmq.ImageHub()
networkFps = FpsCatcher()

while True:  # show streamed images until Ctrl-C
    networkFps.compute()
    rpi_name, frame = image_hub.recv_image()

    frame = cv2.putText(frame, f'Network : {networkFps.fps} fps',
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        cv2.LINE_AA)

    cv2.imshow(rpi_name, frame)  # 1 window for each RPi
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')






