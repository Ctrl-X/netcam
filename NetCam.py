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
                 isstereocam=False,
                 source='0', fullscreen=False, consolelog=True):

        self.captureResolution = capture
        self.displayResolution = display
        self.isStereoCam = isstereocam
        self.source = source
        self.imgWidth, self.imgHeight = resolutionFinder(self.captureResolution, self.isStereoCam)
        self.console(self.imgWidth)

        if self.displayResolution:
            self.displayWidth, self.displayHeight = resolutionFinder(self.displayResolution)
        else:
            self.displayWidth: None

        self.fps = NetCam.MAX_FPS
        self.imgBuffer = [None]
        self.consoleLog = consolelog
        self.isCaptureRunning = False
        self.isDisplayRunning = False
        self.isNetworkRunning = False
        self.fullScreen = fullscreen
        self.videoStream = None

        ## Debug informations
        self.serverIp = serverip
        self.serverPort = serverport
        self.displayDebug = False
        self.showStereo = False
        self.displayFps = FpsCatcher()
        self.captureFps = FpsCatcher()
        self.networkFps = FpsCatcher()

        ## Server Information
        self.threadList = []

        # Start the capture
        self.console('Starting NetCam...')
        self.startCapture()
        time.sleep(0.1)

        ## Launch the display (main thread)
        if self.displayResolution:
            self.startDisplay()

    def startClient(self):
        """
            Launch the network client ( broadcast the camera signal)
        """

        ## Launch the networdThread
        self.console('Init network (client)...', 1)
        zmqContext = zmq.Context()
        socket = zmqContext.socket(zmq.PUB)
        workerThread = Thread(target=self.clientThreadRunner, args=([socket]))
        self.threadList.append(workerThread)
        workerThread.start()
        time.sleep(0.1)
        self.console('NetCam Client started !')

    def clientThreadRunner(self, socket):
        """
            Publish Data to any connected Server
        :param socket:
        """
        url_publish = "tcp://*:%s" % NetCam.DEFAULT_CLIENT_PORT
        self.console(f'Client publishing video on {url_publish}', 2)
        socket.bind(url_publish)
        self.isNetworkRunning = True
        self.console('Network thread is now running ( ZMQ Publish )...', 2)

        i = 0
        topic = 1234
        while self.isNetworkRunning:
            if self.displayDebug:
                self.networkFps.compute()
            # socket.send(self.imgBuffer)
            socket.send("%d %d" % (topic, i))
            time.sleep(0.001)
        self.console('Network thread stopped.', 1)

    def startServer(self):
        """
             Launch the network client ( broadcast the camera signal)
        """
        ## Launch the networdThread
        self.console('Init network (server)...', 1)

        zmqContext = zmq.Context()
        socket = zmqContext.socket(zmq.SUB)
        workerThread = Thread(target=self.serverThreadRunner, args=([socket]))
        self.threadList.append(workerThread)
        workerThread.start()
        time.sleep(0.1)
        self.console('NetCam Server started !')
        # # Socket to talk to clients
        # self.clients = zmqContext.socket(zmq.ROUTER)
        # self.clients.bind(f'tcp://*:{NetCam.DEFAULT_SERVER_PORT}')
        #
        # # Socket to talk to workers
        # self.workers = zmqContext.socket(zmq.DEALER)
        #
        # # Launch pool of worker threads
        # url_worker = "inproc://workers"
        # for i in range(5):
        #     workerThread = Thread(target=self.connectionListener, args=(url_worker,zmqContext))
        #     self.threadList.append(workerThread)
        #     workerThread.start()
        #
        # zmq.device(zmq.QUEUE, self.clients, self.workers)

    def serverThreadRunner(self, socket):
        url_publisher = f"tcp://192.168.0.70:{NetCam.DEFAULT_CLIENT_PORT}"
        socket.connect(url_publisher)
        self.isNetworkRunning = True

        topicfilter = "1234"
        socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

        self.console(f'Connected To {url_publisher}')
        self.console('self.isNetworkRunning', self.isNetworkRunning)
        while self.isNetworkRunning:
            result = socket.recv()
            topic, messagedata = result.split()
            self.console('received', messagedata)
            time.sleep(000.1)
        self.console('Network thread stopped.', 1)

    # def connectionListener2(self, workerUrl, zmqContext = None):
    #     """Worker routine"""
    #     # Context to get inherited or create a new one
    #     zmqContext = zmqContext or zmq.Context.instance()
    #
    #     # Socket to talk to dispatcher
    #     socket = zmqContext.socket(zmq.REP)
    #     socket.connect(workerUrl)
    #
    #     while self.isNetworkRunning:
    #         if self.displayDebug:
    #             self.captureFps.compute()
    #         self.imgBuffer = socket.recv_string()
    #         # self.console("Received request: [ %s ]" % (string))
    #         time.sleep(0.001)
    #         socket.send(b"ACK")

    def startCapture(self):
        """
            Start capturing video frame and put them in the imgBuffer
        """
        ## Close any previously opened stream
        if self.isCaptureRunning and self.videoStream:
            self.videoStream.release()
            self.isCaptureRunning = False

        ## Launch the camera capture thread
        self.console('Init camera capture...', 1)

        ## prepare the triple buffering
        self.videoStream = self.initVideoStream(self.source)

        ## Get the real width, height and fps supported by the camera
        self.imgWidth = int(self.videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.imgHeight = int(self.videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.videoStream.get(cv2.CAP_PROP_FPS)
        self.computeDisplayHeight()
        self.console(f'Capture resolution : {self.imgWidth} x {self.imgHeight} @ {self.fps}', 2)
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
        isOpened = videoStream.isOpened()
        assert isOpened, 'Unable to open camera %s . Is your camera connected (look for videoX in /dev/ ? ' % source

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
        self.isCaptureRunning = True
        self.console('Capture thread is now running.', 2)
        n = 0
        while self.isCaptureRunning:
            n += 1
            # stream.grab()
            if n == 2:
                stream.read(self.imgBuffer)
                n = 0
                if self.displayDebug:
                    self.captureFps.compute()

            time.sleep(0.001)

        if self.videoStream and self.videoStream.isOpened():
            self.videoStream.release()
            self.console('Released camera.', 1)

        self.videoStream = None
        self.console('Capture thread stopped.', 1)

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
            'isCaptureRunning': self.isCaptureRunning,
        })

    def startDisplay(self):
        self.console('Init display...', 1)
        self.console(f'Display resolution : {self.displayResolution} ({self.displayWidth} x {self.displayHeight})', 2)
        cv2.namedWindow(NetCam.DEFAULT_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
        self.toggleFullScreen(self.fullScreen)
        self.isDisplayRunning = True
        time.sleep(0.1)
        self.console('Display is now ready.', 2)

    def setDisplayResolution(self, resolution):
        if (resolution != None):
            self.displayResolution = resolution
            self.displayWidth, _ = resolutionFinder(resolution)
            self.computeDisplayHeight()
            cv2.resizeWindow(NetCam.DEFAULT_WINDOW_NAME, self.displayWidth, self.displayHeight)
            self.console(f'Changed Display resolution for : {resolution} ({self.displayWidth} x {self.displayHeight})')

    def toggleFullScreen(self, isFullScreen=None):
        self.fullScreen = isFullScreen if isFullScreen is not None else not self.fullScreen
        if self.fullScreen:
            self.console(f'Toggle fullscreen')
            cv2.namedWindow(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1.0)
        else:
            cv2.namedWindow(NetCam.DEFAULT_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(NetCam.DEFAULT_WINDOW_NAME, self.displayWidth, self.displayHeight)
            cv2.setWindowProperty(NetCam.DEFAULT_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0.0)

    def toggleDisplayStereo(self, isShowStereo=None):
        self.showStereo = isShowStereo if isShowStereo is not None else not self.showStereo
        self.console(f'Show Stereo : {self.showStereo}')

    def display(self):

        if not self.displayResolution:
            # No Display was setup
            self.console('You need to setup the display Resolution in NetCam constructor. ex : NetCam(display=\'VGA\'')
            time(1)
            return
        if not self.isDisplayRunning:
            cv2.destroyAllWindows()
            return
        # Try to see if the window has been closed by clicking on the right upper cross
        try:
            isWindowClosed = cv2.getWindowProperty(NetCam.DEFAULT_WINDOW_NAME, 0)
            if isWindowClosed == -1:
                # the window has been closed
                self.console("Window was closed.")
                self.clearAll()
        except:
            self.console("Window was closed.")
            self.clearAll()
            return

        frame = self.imgBuffer
        if self.isStereoCam and not self.showStereo:
            # the Display is not in stereo, so remove the half of the picture
            frame = frame[0:self.imgHeight, 0:self.imgWidth // 2]

        if self.displayHeight != self.imgHeight:
            # Resize the picture for display purpose
            width = self.displayWidth if not self.showStereo else self.displayWidth * 2
            frame = cv2.resize(frame, (width, self.displayHeight))

        if self.displayDebug:
            self.displayFps.compute()
            debugTextSize = self.displayWidth / 1280
            thickness = 1 if self.displayWidth < 1280 else 2

            textPosX, textPosY = NetCam.TEXT_POSITION
            textPosX += int(40 * debugTextSize)
            textPosY += int(40 * debugTextSize)
            frame = cv2.putText(frame, f'Capture : {self.captureFps.fps} fps ({self.captureResolution}) | '
                                       f'Display : {self.displayFps.fps} fps ({self.displayResolution}) | '
                                       f'Network : {self.networkFps.fps} fps',
                                (textPosX, textPosY),
                                cv2.FONT_HERSHEY_SIMPLEX, debugTextSize, NetCam.TEXT_COLOR, thickness,
                                cv2.LINE_AA)
            textPosY += int(40 * debugTextSize)
            frame = cv2.putText(frame, f'f : fullscreen | s : see stereo | F1 to F5 : change display',
                                (textPosX, textPosY), cv2.FONT_HERSHEY_SIMPLEX, debugTextSize, NetCam.TEXT_COLOR,
                                thickness,
                                cv2.LINE_AA)

        cv2.imshow(NetCam.DEFAULT_WINDOW_NAME, frame)
        self.listenKeyboard()

    def listenKeyboard(self):
        key = cv2.waitKey(20)
        if key != -1:
            if key == ord('q'):  # q to quit
                self.clearAll()
            elif key == 35:  # Tilde to show debug
                self.toggleDebug()
            elif key == 190:  # F1
                self.setDisplayResolution('QVGA')
            elif key == 191:  # F2
                self.setDisplayResolution('VGA')
            elif key == 192:  # F3
                self.setDisplayResolution('HD')
            elif key == 193:  # F4
                self.setDisplayResolution('FHD')
            elif key == 194:  # F5
                self.setDisplayResolution('2K')
            elif key == ord('f'):  # F to toggle fullscreen
                self.toggleFullScreen()
            elif key == ord('s'):  # S to toggle display stereo
                self.toggleDisplayStereo()
            elif key == 27:  # Esc key was pressed,
                self.toggleFullScreen(False)
            else:
                print(f'Key pressed: {key}')

    def toggleDebug(self):
        self.displayDebug = not self.displayDebug
        self.console(f'Debugging is now {self.displayDebug}.')
        self.displayFps.initTime()
        self.captureFps.initTime()
        self.networkFps.initTime()

    def clearAll(self):
        if self.isNetworkRunning:
            self.console('Stopping Network...')
            self.isNetworkRunning = False
        time.sleep(0.1)
        if self.isDisplayRunning:
            self.console('Stopping Display...')
            self.isDisplayRunning = False
        time.sleep(0.1)
        if self.isCaptureRunning:
            self.console('Stopping Capture...')
            self.isCaptureRunning = False
        time.sleep(0.1)

        self.threadList = []
        zmqContext = zmq.Context.instance()
        zmqContext.term()
        time.sleep(1)
        self.console('Stopping Done.', 1)

    def computeDisplayHeight(self):
        widthMultiplier = 2 if self.isStereoCam else 1
        self.displayHeight = int(self.displayWidth / (self.imgWidth // widthMultiplier) * self.imgHeight)

    def isRunning(self):
        return self.isCaptureRunning or self.isDisplayRunning or self.isNetworkRunning

    def console(self, text, indentlevel=0):
        if self.consoleLog:
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
