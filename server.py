from NetCam import *

if __name__ == "__main__":
    netCam = NetCam(display='FHD', isstereocam=True, flipvertical= True)
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
