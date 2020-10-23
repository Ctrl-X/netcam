from NetCam import *

if __name__ == "__main__":
    netCam = NetCam()
    netCam.startServer()

    try:
        while netCam.isRunning():
            key = cv2.waitKey(1)
            if key != -1:
                if key == ord('q'):  # q to quit
                    netCam.clearAll()
                elif key == 35:  # Tilde to show debug
                    netCam.toggleDebug()
            # netCam.display()

    except KeyboardInterrupt:
        netCam.clearAll()
    except Exception as err:
        netCam.clearAll()
        raise err
    exit()
