
import zmq





if __name__ == "__main__":
    netCam = NetCam()
    netCam.initServer()

    while(netCam.isRunning):
        netCam.display()

