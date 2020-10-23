# run this program on each RPi to send a labelled image stream
import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq

sender = imagezmq.ImageSender(connect_to='tcp://192.168.0.56:5555')

rpi_name = socket.gethostname() # send RPi hostname with each image
picam = VideoStream()

## Define all video settings
# picam.set(cv2.CAP_PROP_BUFFERSIZE, NetCam.NBR_BUFFER)  # increase camera buffering to 3 for triple buffering
# picam.set(cv2.CAP_PROP_FPS, NetCam.MAX_FPS)  # try to put the fps to MAX_FPS
# picam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # define the compression to mjpg
picam.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
picam.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
picam.start()
time.sleep(2.0)  # allow camera sensor to warm up
while True:  # send images as stream until Ctrl-C
    image = picam.read()
    sender.send_image(rpi_name, image)