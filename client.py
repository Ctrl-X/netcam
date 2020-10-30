## Main.py is only use to run the NetCam class, nothing particulery interesting inside...

from NetCam import *

if __name__ == '__main__':
    print('Started main.py...')
    netCam = NetCam(display=None, capture='VGA', isStereoCam=False)
    netCam.startBroadcast()

    try:
        while netCam.isRunning():
            netCam.display()


    except KeyboardInterrupt:
        netCam.clearAll()
    except Exception as err:
        netCam.clearAll()
        raise err
    exit()
