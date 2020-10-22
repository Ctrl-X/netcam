from NetCam import *

if __name__ == "__main__":
    netCam = NetCam()
    netCam.startServer()

    while netCam.isRunning:
        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('q'):  # q to quit
                netCam.clearAll()
        # netCam.display()

    exit()
