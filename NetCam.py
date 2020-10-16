################################################################################
##          Classe NetCam
##      Gere un flux camera caméra à travers un réseau
##  Author : J. Coupez
##  Date : 10/10/2020
##  Version : 0.1
################################################################################

import time
from threading import Thread

import cv2
import numpy as np



class NetCam:
    DEFAULT_IP = '10.10.64.154'
    DEFAULT_PORT = '5555'

    DEFAULT_RES = 'HD'
    MAX_FPS = 60
    NBR_BUFFER = 3

    TEXT_COLOR = (0, 0, 255)
    TEXT_POSITION = (20, 20)

    def __init__(self, serverip=DEFAULT_IP, serverport=DEFAULT_PORT, resolution=DEFAULT_RES, isstereocam=True,
                 source='0', fullscreen=False):
        self.serverIp = serverip
        self.serverPort = serverport
        self.resolution = resolution
        self.isStereoCam = isstereocam
        self.source = source
        self.imgWidth, self.imgHeight = resolutionFinder(self.resolution, self.isStereoCam)
        self.displayResolution = resolution
        self.displayWidth = self.imgWidth
        self.displayHeight = self.imgHeight
        self.fps = NetCam.MAX_FPS
        self.imgBuffer = [None]
        self.isRunning = False
        self.fullScreen = fullscreen

        ## Debug informations
        self.displayDebug = False
        self.displayFps = FpsCatcher()
        self.captureFps = FpsCatcher()


    def startCapture(self):
        """
            Start capturing video frame and put them in the imgBuffer
        """
        ## Close any previously opened stream
        if self.isRunning:
            self.videoStream.release()

        ## prepare the triple buffering
        self.videoStream = self.initVideoStream(self.source)

        ## Get the real width, height and fps supported by the camera
        self.imgWidth = int(self.videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.imgHeight = int(self.videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.videoStream.get(cv2.CAP_PROP_FPS)
        self.imgBuffer = np.empty(shape=(self.imgHeight, self.imgWidth, 3), dtype=np.uint8)

        ## Guarantee the first frame
        self.videoStream.read(self.imgBuffer)

        ## Launch the capture thread
        thread = Thread(target=self.update, args=([self.videoStream]), daemon=True)
        thread.start()

    def stopCapture(self):
        if self.videoStream and self.videoStream.isOpened():
            self.videoStream.release()
        self.isRunning = False

    def initVideoStream(self, source):
        """
            Initialize the video stream with the right resolution and settings
        :param source: the name of the camera device to use for capture. use video0 if not provided
        """

        videoStream = cv2.VideoCapture(0 if source == '0' else source, cv2.CAP_V4L2)
        self.isRunning = videoStream.isOpened()
        assert self.isRunning, 'Unable to open camera %s . Is your camera connected (look for videoX in /dev/ ? ' % source

        ## Get the requested resolution
        width, height = resolutionFinder(self.resolution, self.isStereoCam)

        ## Define all video settings
        videoStream.set(cv2.CAP_PROP_BUFFERSIZE,
                        NetCam.NBR_BUFFER)  # increase camera buffering to 3 for triple buffering
        videoStream.set(cv2.CAP_PROP_FPS, NetCam.MAX_FPS)  # try to put the fps to MAX_FPS
        videoStream.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # define the compression to mjpg
        videoStream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        videoStream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return videoStream

    def update(self, stream):
        """
            Read next stream frame in a daemon thread
        :param stream: videoStream to read from
        """
        #n = 0
        while stream.isOpened():
            #n += 1
            stream.grab()
            stream.retrieve(self.imgBuffer)

            if self.displayDebug:
                self.captureFps.compute()
            #time.sleep(0.01)
        self.isRunning = False

    def getDetail(self):
        return ({
            'serverIp': self.serverIp,
            'serverPort': self.serverPort,
            'resolution': self.resolution,
            'isStereo': self.isStereoCam,
            'width': self.imgWidth,
            'height': self.imgHeight,
            'height': self.imgHeight,
            'maxFps': self.fps,
            'isRunning': self.isRunning,
        })

    def setDisplayResolution(self,resolution):
        if (resolution != None):
            self.displayResolution = resolution
            self.displayWidth, self.displayHeight = resolutionFinder(resolution, self.isStereoCam)


    def toggleFullSreen(self):
        self.fullScreen = not self.fullScreen


    def display(self):
        frame = self.imgBuffer
        if self.displayWidth != self.imgWidth:
            frame = cv2.resize(frame, (self.displayWidth, self.displayHeight))
        if self.displayDebug:
            self.displayFps.compute()
            textPosX, textPosY = NetCam.TEXT_POSITION
            frame = cv2.putText(frame, f'Display : {self.displayFps.fps} fps ({self.displayResolution})', (textPosX,textPosY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NetCam.TEXT_COLOR, 1,
                                cv2.LINE_AA)
            textPosY += 20
            frame = cv2.putText(frame, f'Capture : {self.captureFps.fps} fps ({self.resolution})', (textPosX,textPosY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NetCam.TEXT_COLOR, 1,
                                cv2.LINE_AA)
            textPosY += 20
            frame = cv2.putText(frame, f'Fullscreen : {self.fullScreen}', (textPosX,textPosY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NetCam.TEXT_COLOR, 1,
                                cv2.LINE_AA)

        if self.fullScreen:
            cv2.namedWindow('stream',cv2.WINDOW_GUI_NORMAL)
            #cv2.namedWindow('stream',cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('stream', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        cv2.imshow('stream', frame)

    def toggleDebug(self):
        self.displayDebug = not self.displayDebug
        self.displayFps.initTime()
        self.captureFps.initTime()


def resolutionFinder(res, isStereoCam):
    widthMultiplier = 2 if isStereoCam else 1
    switcher = {
        'QVGA': (320 * widthMultiplier, 240),
        'VGA': (640 * widthMultiplier, 480),
        'HD': (1280 * widthMultiplier, 720),
        'FHD': (1920 * widthMultiplier, 1080),
        '2K': (2048 * widthMultiplier, 1080)
    }
    return switcher.get(res, (640 * widthMultiplier, 480))




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
