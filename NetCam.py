################################################################################
##          Classe NetCam
##      Gere un flux camera caméra à travers un réseau
##  Author : J. Coupez
##  Date : 10/10/2020
##  Version : 0.1
################################################################################

import time
from threading import Thread
import zmq
import cv2
import numpy as np



class NetCam:
    DEFAULT_IP = '10.10.64.154'
    DEFAULT_SERVER_PORT = '5555'
    DEFAULT_CLIENT_PORT = '5556'

    DEFAULT_RES = 'HD'
    MAX_FPS = 60
    NBR_BUFFER = 3

    TEXT_COLOR = (0, 0, 255)
    TEXT_POSITION = (20, 20)

    def __init__(self, serverip=DEFAULT_IP, serverport=DEFAULT_SERVER_PORT, resolution=DEFAULT_RES, isstereocam=True,
                 source='0', fullscreen=False):

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
        self.videothread = None

        ## Debug informations
        self.serverIp = serverip
        self.serverPort = serverport
        self.displayDebug = False
        self.displayFps = FpsCatcher()
        self.captureFps = FpsCatcher()

        ## Server Information
        self.workerThread = []
        self.workers = None
        self.clients = None


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
        self.videothread = Thread(target=self.update, args=([self.videoStream]), daemon=True)
        self.videothread.start()

    def stopCapture(self):
        if self.videothread != None:
            self.videothread.stop()
        if self.videoStream and self.videoStream.isOpened():
            self.videoStream.release()
        self.isRunning = False

    def startClient(self):
        """
            Launch the network client
        """
        zmqContext = zmq.Context()
        socket = zmqContext.socket(zmq.PUB)
        workerThread = Thread(target=self.publish, args=(socket))
        self.workerThread.append(workerThread)
        self.isRunning = True
        workerThread.start()


    def publish(self, socket):
        """
            Publish Data to any connected Server
        :param socket:
        """
        socket.bind("tcp://*:%s" % NetCam.DEFAULT_CLIENT_PORT)
        while self.isRunning:
            socket.send(self.imgBuffer)
            time.sleep(1)


    def startServer(self):
        # Prepare our context and sockets
        zmqContext = zmq.Context.instance()

        # Socket to talk to clients
        self.clients = zmqContext.socket(zmq.ROUTER)
        self.clients.bind(f'tcp://*:{NetCam.DEFAULT_SERVER_PORT}')

        # Socket to talk to workers
        self.workers = zmqContext.socket(zmq.DEALER)

        # Launch pool of worker threads
        url_worker = "inproc://workers"
        self.isRunning = True
        for i in range(5):
            workerThread = Thread(target=self.connectionListener, args=(url_worker,zmqContext))
            self.workerThread.append(workerThread)
            workerThread.start()

        zmq.device(zmq.QUEUE, self.clients, self.workers)

    def stopServer(self):
        self.isRunning = False
        if self.workerThread:
            for worker in self.workerThread:
                worker.stop()
        if self.clients != None:
            self.clients.close()
        if self.workers != None:
            self.workers.close()
        zmqContext = zmq.Context.instance()
        zmqContext.term()


    def connectionListener(self, workerUrl, zmqContext = None):
        """Worker routine"""
        # Context to get inherited or create a new one
        zmqContext = zmqContext or zmq.Context.instance()

        # Socket to talk to dispatcher
        socket = zmqContext.socket(zmq.REP)
        socket.connect(workerUrl)

        while self.isRunning:
            if self.displayDebug:
                self.captureFps.compute()
            self.imgBuffer = socket.recv_string()
            # print("Received request: [ %s ]" % (string))
            time.sleep(1)
            socket.send(b"ACK")




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
