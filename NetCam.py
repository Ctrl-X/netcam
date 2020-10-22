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
    DEFAULT_WINDOW_NAME = 'Stream'

    DEFAULT_RES = 'HD'
    MAX_FPS = 60
    NBR_BUFFER = 3

    TEXT_COLOR = (0, 0, 255)
    TEXT_POSITION = (0, 0)

    def __init__(self, serverip=DEFAULT_IP, serverport=DEFAULT_SERVER_PORT, capture=DEFAULT_RES, display=None,
                 isstereocam=True,
                 source='0', fullscreen=False):

        self.captureResolution = capture
        self.displayResolution = display
        self.isStereoCam = isstereocam
        self.source = source
        self.imgWidth, self.imgHeight = resolutionFinder(self.captureResolution, self.isStereoCam)
        console(self.imgWidth)

        if self.displayResolution:
            self.displayWidth, self.displayHeight = resolutionFinder(self.displayResolution)
        else:
            self.displayWidth: None

        self.fps = NetCam.MAX_FPS
        self.imgBuffer = [None]
        self.isRunning = False
        self.fullScreen = fullscreen
        self.videoStream = None

        ## Debug informations
        self.serverIp = serverip
        self.serverPort = serverport
        self.displayDebug = False
        self.displayFps = FpsCatcher()
        self.captureFps = FpsCatcher()
        self.networkFps = FpsCatcher()

        ## Server Information
        self.threadList = []

    def startClient(self):
        """
            Launch the network client ( broadcast the camera signal)
        """
        console('Starting NetCam Client...')
        self.isRunning = True
        ## Launch the camera capture thread
        console('Init camera capture...', 1)
        self.startCapture()
        time.sleep(0.1)

        ## Launch the networdThread
        console('Init network...', 1)
        zmqContext = zmq.Context()
        socket = zmqContext.socket(zmq.PUB)
        workerThread = Thread(target=self.clientThreadRunner, args=([socket]))
        self.threadList.append(workerThread)
        workerThread.start()
        time.sleep(0.1)

        ## Launch the display Thread (main thread)
        if self.displayResolution:
            console('Init display...', 1)
            console(f'Display resolution : {self.displayResolution} ({self.displayWidth} x {self.displayHeight})', 2)
            cv2.namedWindow(NetCam.DEFAULT_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
            self.toggleFullScreen(self.fullScreen)
            time.sleep(0.1)
            console('Display is now ready.', 2)

        console('NetCam Client started !')

    def clientThreadRunner(self, socket):
        """
            Publish Data to any connected Server
        :param socket:
        """
        url_publish = "tcp://*:%s" % NetCam.DEFAULT_CLIENT_PORT
        console(f'Publishing video on {url_publish}', 2)
        socket.bind(url_publish)
        console('Network thread is now running ( ZMQ Publish)...', 2)

        i = 0
        while self.isRunning:
            if self.displayDebug:
                self.networkFps.compute()
            socket.send_string(f'{i}')
            time.sleep(0.001)
        console('Network thread stopped.', 1)

    def startServer(self):
        """
             Launch the network client ( broadcast the camera signal)
        """
        self.isRunning = True

        ## Launch the networdThread
        zmqContext = zmq.Context()
        socket = zmqContext.socket(zmq.SUB)
        workerThread = Thread(target=self.serverThreadRunner, args=([socket]))
        self.threadList.append(workerThread)
        workerThread.start()

        # # Socket to talk to clients
        # self.clients = zmqContext.socket(zmq.ROUTER)
        # self.clients.bind(f'tcp://*:{NetCam.DEFAULT_SERVER_PORT}')
        #
        # # Socket to talk to workers
        # self.workers = zmqContext.socket(zmq.DEALER)
        #
        # # Launch pool of worker threads
        # url_worker = "inproc://workers"
        # self.isRunning = True
        # for i in range(5):
        #     workerThread = Thread(target=self.connectionListener, args=(url_worker,zmqContext))
        #     self.threadList.append(workerThread)
        #     workerThread.start()
        #
        # zmq.device(zmq.QUEUE, self.clients, self.workers)

    def serverThreadRunner(self, socket):
        url_publisher = f"tcp://192.168.1.246:{NetCam.DEFAULT_CLIENT_PORT}"
        socket.connect(url_publisher)

        console(f'Connected To {url_publisher}')
        console('self.isRunning', self.isRunning)
        while self.isRunning:
            result = socket.recv_string()
            console('received', result)
            time.sleep(000.1)

    # def connectionListener2(self, workerUrl, zmqContext = None):
    #     """Worker routine"""
    #     # Context to get inherited or create a new one
    #     zmqContext = zmqContext or zmq.Context.instance()
    #
    #     # Socket to talk to dispatcher
    #     socket = zmqContext.socket(zmq.REP)
    #     socket.connect(workerUrl)
    #
    #     while self.isRunning:
    #         if self.displayDebug:
    #             self.captureFps.compute()
    #         self.imgBuffer = socket.recv_string()
    #         # console("Received request: [ %s ]" % (string))
    #         time.sleep(0.001)
    #         socket.send(b"ACK")

    def startCapture(self):
        """
            Start capturing video frame and put them in the imgBuffer
        """
        ## Close any previously opened stream
        if self.isRunning and self.videoStream:
            self.videoStream.release()

        ## prepare the triple buffering
        self.videoStream = self.initVideoStream(self.source)

        ## Get the real width, height and fps supported by the camera
        self.imgWidth = int(self.videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.imgHeight = int(self.videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.computeDisplayHeight()
        self.fps = self.videoStream.get(cv2.CAP_PROP_FPS)
        console(f'Capture resolution : {self.imgWidth} x {self.imgHeight} @ {self.fps}', 2)
        self.imgBuffer = np.empty(shape=(self.imgHeight, self.imgWidth, 3), dtype=np.uint8)

        ## Guarantee the first frame
        self.videoStream.read(self.imgBuffer)

        ## Launch the capture thread
        videoThread = Thread(target=self.captureThreadRunner, args=([self.videoStream]), daemon=True)
        videoThread.start()

    def initVideoStream(self, source):
        """
            Initialize the video stream with the right resolution and settings
        :param source: the name of the camera device to use for capture. use video0 if not provided
        """

        videoStream = cv2.VideoCapture(0 if source == '0' else source, cv2.CAP_V4L2)
        self.isRunning = videoStream.isOpened()
        assert self.isRunning, 'Unable to open camera %s . Is your camera connected (look for videoX in /dev/ ? ' % source

        ## Get the requested resolution
        width, height = resolutionFinder(self.captureResolution, self.isStereoCam)

        ## Define all video settings
        videoStream.set(cv2.CAP_PROP_BUFFERSIZE,
                        NetCam.NBR_BUFFER)  # increase camera buffering to 3 for triple buffering
        videoStream.set(cv2.CAP_PROP_FPS, NetCam.MAX_FPS)  # try to put the fps to MAX_FPS
        videoStream.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # define the compression to mjpg
        videoStream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        videoStream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return videoStream

    def captureThreadRunner(self, stream):
        """
            Read next stream frame in a daemon thread
        :param stream: videoStream to read from
        """
        console('Capture thread is now running.', 2)
        # n = 0
        while self.isRunning and stream.isOpened():
            # n += 1
            stream.grab()
            stream.retrieve(self.imgBuffer)

            if self.displayDebug:
                self.captureFps.compute()
            time.sleep(0.001)

        if self.videoStream and self.videoStream.isOpened():
            self.videoStream.release()
            console('Released camera.', 1)

        self.videoStream = None
        console('Capture thread stopped.', 1)

    def getDetail(self):
        return ({
            'serverIp': self.serverIp,
            'serverPort': self.serverPort,
            'captureResolution': self.captureResolution,
            'displayResolution': self.displayResolution,
            'isStereo': self.isStereoCam,
            'width': self.imgWidth,
            'height': self.imgHeight,
            'maxFps': self.fps,
            'isRunning': self.isRunning,
        })

    def setDisplayResolution(self, resolution):
        if (resolution != None):
            self.displayResolution = resolution
            self.displayWidth, _ = resolutionFinder(resolution)
            self.computeDisplayHeight()
            cv2.resizeWindow(NetCam.DEFAULT_WINDOW_NAME, self.displayWidth, self.displayHeight)
            console(f'Changed Display resolution for : {resolution} ({self.displayWidth} x {self.displayHeight})')

    def toggleFullScreen(self, isFullScreen=None):
        self.fullScreen = isFullScreen if isFullScreen is not None else not self.fullScreen
        console(f'Toggle fullscreen : {self.fullScreen}')
        if self.fullScreen:
            cv2.namedWindow(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1.0)
        else:
            cv2.namedWindow(NetCam.DEFAULT_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(NetCam.DEFAULT_WINDOW_NAME, self.displayWidth, self.displayHeight)
            cv2.setWindowProperty(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0.0)

    def display(self):

        if not self.displayResolution:
            # No Display was setup
            console('You need to setup the display Resolution in NetCam constructor. ex : NetCam(display=\'VGA\'')
            time(1)
            return
        if not self.isRunning:
            cv2.destroyAllWindows()
            return
        try:
            isWindowClosed = cv2.getWindowProperty(NetCam.DEFAULT_WINDOW_NAME, 0)
            if isWindowClosed == -1:
                # the window has been closed
                console("Window was closed.")
                self.clearAll()
        except:
            console("Window was closed.")
            self.clearAll()
            return

        frame = self.imgBuffer
        if self.isStereoCam:
            # the Display is not in stereo, so remove the half of the picture
            frame = frame[0:self.imgHeight, 0:self.imgWidth // 2]

        if self.displayHeight != self.imgHeight:
            # Resize the picture for display purpose
            frame = cv2.resize(frame, (self.displayWidth, self.displayHeight))

        if self.displayDebug:
            self.displayFps.compute()
            debugTextSize = self.displayWidth / 1280
            thickness = 1 if self.displayWidth < 1280 else 2

            textPosX, textPosY = NetCam.TEXT_POSITION
            textPosX += int(40 * debugTextSize)
            textPosY += int(40 * debugTextSize)
            frame = cv2.putText(frame, f'Capture : {self.captureFps.fps} fps ({self.captureResolution})\r\n'
                                       f'Display : {self.displayFps.fps} fps ({self.displayResolution})\n'
                                       f'Network : {self.networkFps.fps} fps\n'
                                       f's : see stereo',
                                (textPosX, textPosY),
                                cv2.FONT_HERSHEY_SIMPLEX, debugTextSize, NetCam.TEXT_COLOR, thickness,
                                cv2.LINE_AA)
           #  textPosY += int(40 * debugTextSize)
           #  frame = cv2.putText(frame, f'Display : {self.displayFps.fps} fps ({self.displayResolution})',
           #                      (textPosX, textPosY), cv2.FONT_HERSHEY_SIMPLEX, debugTextSize, NetCam.TEXT_COLOR,
           #                      thickness,
           #                      cv2.LINE_AA)
           #  textPosY += int(40 * debugTextSize)
           #  frame = cv2.putText(frame, f'Network : {self.networkFps.fps} fps', (textPosX, textPosY),
           #                      cv2.FONT_HERSHEY_SIMPLEX, debugTextSize, NetCam.TEXT_COLOR, thickness,
           #                      cv2.LINE_AA)
           # textPosY += int(40 * debugTextSize)
           # frame = cv2.putText(frame, f's : see stereo', (textPosX, textPosY),
           #                      cv2.FONT_HERSHEY_SIMPLEX, debugTextSize, NetCam.TEXT_COLOR, thickness,
           #                      cv2.LINE_AA)
        cv2.imshow(NetCam.DEFAULT_WINDOW_NAME, frame)

    def toggleDebug(self):
        self.displayDebug = not self.displayDebug
        console(f'Debugging is now {self.displayDebug}.')
        self.displayFps.initTime()
        self.captureFps.initTime()

    def clearAll(self):
        if self.isRunning:
            console('Stopping NetCam...')
            self.isRunning = False
            self.threadList = []
            zmqContext = zmq.Context.instance()
            zmqContext.term()
            time.sleep(1)
            console('Stopping Done.', 1)

    def computeDisplayHeight(self):
        widthMultiplier = 2 if self.isStereoCam else 1
        self.displayHeight = int(self.displayWidth / (self.imgWidth // widthMultiplier) * self.imgHeight)


def console(text, indentlevel=0):
    output = ''
    for count in range(0, indentlevel):
        output = output + '\t'
    output = output + time.strftime('%b %d at %l:%M:%S')
    print(f'{output} : {text}')


def resolutionFinder(res, isstereocam=False):
    widthMultiplier = 2 if isstereocam else 1
    # switcher = {
    #     'QVGA': 320 ,
    #     'VGA': 640 ,
    #     'HD': 1280 ,
    #     'FHD': 1920 ,
    #     '2K': 2048
    # }
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
