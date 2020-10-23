from NetCam import *

if __name__ == "__main__":
    netCam = NetCam(display='VGA', isstereocam=True)
    netCam.startServer()

    try:
        while netCam.isRunning():
            netCam.display()

    except KeyboardInterrupt:
        netCam.clearAll()
    except Exception as err:
        netCam.clearAll()
        raise err
    exit()
