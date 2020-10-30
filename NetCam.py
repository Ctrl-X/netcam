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
import base64
import numpy as np
import socket


class NetCam:
    DEFAULT_CLIENT_PORT = '5556'
    DEFAULT_WINDOW_NAME = 'Stream'
    WINDOW_COUNTER = 0

    DEFAULT_RES = 'HD'
    MAX_FPS = 60
    NBR_BUFFER = 5

    TEXT_COLOR = (0, 0, 255)
    TEXT_POSITION = (0, 0)

    def __init__(self,
                 capture='VGA',
                 display=None,
                 isStereoCam=False,
                 source='0',
                 ip=None,
                 port=None,
                 consolelog=True):

        self.consoleLog = consolelog
        self.captureResolution = capture
        self.displayResolution = display
        self.isStereoCam = isStereoCam
        self.source = source
        self.imgWidth, self.imgHeight = resolutionFinder(self.captureResolution, self.isStereoCam)
        self.displayWidth, self.displayHeight = resolutionFinder(self.displayResolution)

        self.fps = NetCam.MAX_FPS
        self.imgBuffer = [None] * NetCam.NBR_BUFFER
        self.imgBufferReady = 0
        self.imgBufferWriting = 0
        self.flipVertical = False
        self.isCaptureRunning = False
        self.isDisplayRunning = False
        self.isNetworkRunning = False
        self.fullScreen = False
        self.videoStream = None

        ## Debug informations
        self.displayDebug = False
        self.showStereo = False
        self.displayFps = FpsCatcher()
        self.captureFps = FpsCatcher()
        self.networkFps = FpsCatcher()

        # Network information
        self.hostname = socket.gethostname()
        self.ip_address = ip
        self.ip_port = port or NetCam.DEFAULT_CLIENT_PORT
        self.windowName = ip or self.hostname or NetCam.DEFAULT_WINDOW_NAME
        NetCam.WINDOW_COUNTER += 1
        if NetCam.WINDOW_COUNTER > 1:
            self.windowName += f' ({NetCam.WINDOW_COUNTER})'


        self.threadList = []

        self.console('Starting NetCam...')

        if self.ip_address is None:
            # Start the capture
            self.startCapture()
        else:
            # Start to receive the stream
            self.startReceive()

        time.sleep(0.1)

        ## Init the display when requested (on main thread)
        if self.displayResolution:
            self.initDisplay()
            time.sleep(0.1)

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
        self.console(f'Capture resolution : {self.imgWidth} x {self.imgHeight} @ {self.fps}', 1)

        # Initialise each buffer
        for i in range(NetCam.NBR_BUFFER):
            self.imgBuffer[i] = np.empty(shape=(self.imgHeight, self.imgWidth, 3), dtype=np.uint8)

        ## Guarantee the first frame
        self.videoStream.read(self.imgBuffer[self.imgBufferWriting])
        self.imgBufferWriting += 1

        ## Launch the capture thread
        videoThread = Thread(target=self.captureThreadRunner, args=([self.videoStream]), daemon=True)
        videoThread.start()

    def startBroadcast(self):
        """
            Launch the network client ( broadcast the camera signal)
        """

        ## Launch the networdThread
        self.ip_address = get_ip()

        self.console(f'Launch broadcast...')
        zmqContext = zmq.Context()
        # zmqContext = SerializingContext()
        socket = zmqContext.socket(zmq.PUB)
        workerThread = Thread(target=self.clientThreadRunner, args=([socket]))
        self.threadList.append(workerThread)
        workerThread.start()
        time.sleep(0.1)
        self.console(f'Now broadcasting. URI of Camera : {self.ip_address}:{self.ip_port} !')

    def startReceive(self):
        """
             Launch the network client ( broadcast the camera signal)
        """
        ## Launch the networdThread
        self.console(f'Connecting to camera on {self.ip_address}:{self.ip_port}...', 1)

        zmqContext = zmq.Context()
        # zmqContext = SerializingContext()
        socket = zmqContext.socket(zmq.SUB)
        workerThread = Thread(target=self.serverThreadRunner, args=([socket]))
        self.threadList.append(workerThread)
        workerThread.start()
        time.sleep(0.1)

    def initVideoStream(self, source):
        """
            Initialize the video stream with the right resolution and settings
        :param source: the name of the camera device to use for capture. use video0 if not provided
        """

        videoStream = cv2.VideoCapture(0 if source == '0' else source, cv2.CAP_V4L2)
        isOpened = videoStream.isOpened()
        if not isOpened:
            # Try to open the camera without the Video for linux driver
            videoStream = cv2.VideoCapture(0 if source == '0' else source)
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
        self.console('Capture thread is now running.', 1)
        while self.isCaptureRunning:
            # For buffering : Never read where we write
            self.imgBufferReady = self.imgBufferWriting
            self.imgBufferWriting = 0 if self.imgBufferWriting == NetCam.NBR_BUFFER - 1 else self.imgBufferWriting + 1

            stream.read(self.imgBuffer[self.imgBufferWriting])
            if self.displayDebug:
                self.captureFps.compute()

            time.sleep(0.001)

        if self.videoStream and self.videoStream.isOpened():
            self.videoStream.release()
            self.console('Released camera.', 1)

        self.videoStream = None
        self.console('Capture thread stopped.', 1)

    def clientThreadRunner(self, socket):
        """
            Publish Data to any connected Server
        :param socket:
        """
        url_publish = "tcp://*:%s" % self.ip_port
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.set_hwm(2)
        socket.bind(url_publish)
        self.isNetworkRunning = True
        self.console(f'Network thread is now running ( {url_publish} )...', 1)

        # i = 0
        # topic = 1234
        initTime = FpsCatcher.currentMilliTime()
        bufferSize = 0
        bufferSizeSec = 0
        frameCount = 0
        while self.isNetworkRunning:
            if self.displayDebug:
                self.networkFps.compute()
            currentTime = FpsCatcher.currentMilliTime()
            # buffer = [1]
            encoded, buffer = cv2.imencode('.jpg', np.empty(shape=(5, 5, 3), dtype=np.uint8))

            bufferSize = len(buffer)/1024
            bufferSizeSec += bufferSize
            frameCount += 1
            self.console(f'buffer size : {bufferSize} ko')

            if currentTime - initTime > 1000:
                self.console(f'frame send per sec: {frameCount}',1)
                self.console(f'buffer size per sec : {bufferSizeSec} ko',1)
                bufferSizeSec = 0
                frameCount = 0
                initTime = currentTime

            socket.send(buffer, copy=False)
            processTime = FpsCatcher.currentMilliTime() - currentTime
            waitTime = 1
            if processTime > 0 and processTime < 33:
                waitTime = 33 - processTime
            # self.console(f'processTime : {processTime} milli - waitTime: {waitTime} milli')
            waitTime = waitTime / 1000.0

            time.sleep(waitTime)
        self.console('Network thread stopped.')

    def serverThreadRunner(self, socket):
        url_publisher = f"tcp://{self.ip_address}:{self.ip_port}"

        # topicfilter = "1234"
        socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
        socket.set_hwm(2)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(url_publisher)
        self.isNetworkRunning = True

        # socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

        self.console(f'Connected To {url_publisher}')
        timeoutMsg = 0
        initTime = FpsCatcher.currentMilliTime()
        bufferSize = 0
        frameCount = 0
        while self.isNetworkRunning:
            try:
                currentTime = FpsCatcher.currentMilliTime()
                buffer = socket.recv(flags=zmq.NOBLOCK, copy=False)
                bufferSize += len(buffer) / 1024
                frameCount += 1
                if currentTime - initTime > 1000:
                    self.console(f'frame send per sec: {frameCount}')
                    self.console(f'buffer size per sec : {bufferSize} ko')
                    bufferSize = 0
                    frameCount = 0
                    initTime = currentTime

                shape = [len(buffer.bytes), 1]

                buffer = np.frombuffer(buffer, dtype='uint8')
                buffer = buffer.reshape(shape)

                # For buffering : Never read where we write
                self.imgBufferReady = self.imgBufferWriting
                self.imgBufferWriting = 0 if self.imgBufferWriting == NetCam.NBR_BUFFER - 1 else self.imgBufferWriting + 1

                self.imgBuffer[self.imgBufferWriting] = cv2.imdecode(buffer, 1)

                if self.displayDebug:
                    self.networkFps.compute()

                if timeoutMsg >= 1000:  # 1 sec elapsed
                    self.console(f'Re-Connected To {url_publisher}')
                timeoutMsg = 0



            except Exception as err:
                timeoutMsg += 1
                if timeoutMsg >= 3000:   # 3 sec before writing timeout message
                    self.console(f'Waiting image from {url_publisher} ...')
                    timeoutMsg = 0

            time.sleep(0.001)

        self.isNetworkRunning = False
        self.console('Network thread stopped.')

    def getDetail(self):
        return ({
            'captureResolution': self.captureResolution,
            'displayResolution': self.displayResolution,
            'isStereo': self.isStereoCam,
            'width': self.imgWidth,
            'height': self.imgHeight,
            'maxFps': self.fps,
            'isCaptureRunning': self.isCaptureRunning,
        })

    def initDisplay(self):
        self.console('Init display...', 1)
        self.console(f'Display resolution : {self.displayResolution} ({self.displayWidth} x {self.displayHeight})', 2)
        cv2.namedWindow(self.windowName, cv2.WINDOW_GUI_NORMAL)
        self.toggleFullScreen(self.fullScreen)
        self.isDisplayRunning = True
        time.sleep(0.1)
        self.console('Display is now ready.', 2)

    def display(self):

        if not self.displayResolution:
            # No Display was setup
            # self.console('You need to setup the display Resolution in NetCam constructor. ex : NetCam(display=\'VGA\'')
            # time(1)
            return
        if not self.isDisplayRunning:
            cv2.destroyAllWindows()
            return
        # Try to see if the window has been closed by clicking on the right upper cross
        try:
            isWindowClosed = cv2.getWindowProperty(self.windowName, 0)
            if isWindowClosed == -1:
                # the window has been closed
                self.console("Window was closed.")
                self.clearAll()
        except:
            self.console("Window was closed.")
            self.clearAll()
            return

        frame = self.imgBuffer[self.imgBufferReady]
        if frame is None:
            return  # Nothing to display

        if self.isStereoCam and not self.showStereo:
            # the Display is not in stereo, so remove the half of the picture
            height, width, _ = frame.shape
            frame = frame[0:height, 0:width // 2]

        if self.displayHeight != self.imgHeight:
            # Resize the picture for display purpose
            width = self.displayWidth if not self.showStereo else self.displayWidth * 2
            frame = cv2.resize(frame, (width, self.displayHeight))
        else:
            frame = np.copy(frame)

        if self.flipVertical:
            frame = cv2.flip(frame, 0)

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

        cv2.imshow(self.windowName, frame)
        self.listenKeyboard()

    def setDisplayResolution(self, resolution):
        if (resolution != None):
            self.displayResolution = resolution
            self.displayWidth, self.displayHeight = resolutionFinder(resolution)
            self.computeDisplayHeight()
            cv2.resizeWindow(self.windowName, self.displayWidth, self.displayHeight)
            self.console(f'Changed Display resolution for : {resolution} ({self.displayWidth} x {self.displayHeight})')

    def toggleFullScreen(self, isFullScreen=None):
        self.fullScreen = isFullScreen if isFullScreen is not None else not self.fullScreen
        if self.fullScreen:
            self.console(f'Toggle fullscreen')
            cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.setWindowProperty(self.windowName, cv2.WND_PROP_TOPMOST, 1.0)
        else:
            cv2.namedWindow(self.windowName, cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.windowName, self.displayWidth, self.displayHeight)
            # cv2.setWindowProperty(self.windowName, cv2.WND_PROP_TOPMOST, 0.0)

    def toggleDisplayStereo(self, isShowStereo=None):
        self.showStereo = isShowStereo if isShowStereo is not None else not self.showStereo
        self.console(f'Show Stereo : {self.showStereo}')

    def listenKeyboard(self):
        key = cv2.waitKey(20)
        if key != -1:
            if key == ord('q'):  # q to quit
                self.clearAll()
            elif key == 35 or key == 47:  # Tilde to show debug
                self.toggleDebug()
            elif key == 190 or key == 122:  # F1
                self.setDisplayResolution('QVGA')
            elif key == 191 or key == 120:  # F2
                self.setDisplayResolution('VGA')
            elif key == 192 or key == 99:  # F3
                self.setDisplayResolution('HD')
            elif key == 193 or key == 118:  # F4
                self.setDisplayResolution('FHD')
            elif key == 194 or key == 96:  # F5
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
        time.sleep(0.5)
        self.console('Stopping Done.')

    def computeDisplayHeight(self):
        widthMultiplier = 2 if self.isStereoCam else 1

        if self.imgWidth and self.imgHeight and self.displayWidth:
            self.displayHeight = int(self.displayWidth / (self.imgWidth // widthMultiplier) * self.imgHeight)

    def invertVertical(self):
        self.flipVertical = not self.flipVertical

    def isRunning(self):
        return self.isCaptureRunning or self.isDisplayRunning or self.isNetworkRunning

    def console(self, text, indentlevel=0):
        if self.consoleLog:
            output = time.strftime('%b %d at %H:%M:%S') + ' : '
            for count in range(0, indentlevel):
                output = output + '\t'
            print(f'{output}{text}')


def resolutionFinder(res, isstereocam=False):
    if res == None:
        return (None, None)
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


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


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
