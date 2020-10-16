## Main.py is only use to run the NetCam class, nothing particulery interesting inside...

from NetCam import *

if __name__ == '__main__':
    print('Started main.py...')
    netCam = NetCam()
    netCam.setDisplayResolution('QVGA')
    print('Camera Info', netCam.getDetail())

    print('Starting capture...')
    netCam.startCapture()
    netCam.startClient()

    while netCam.isRunning:

        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('q'):  # q to quit
                cv2.destroyAllWindows()
                raise StopIteration
            elif key == ord('d'):  # d to show debug
                netCam.toggleDebug()
            elif key == 190:  # F1
                netCam.setDisplayResolution('QVGA')
            elif key == 191:  # F2
                netCam.setDisplayResolution('VGA')
            elif key == 192:  # F3
                netCam.setDisplayResolution('HD')
            elif key == 193:  # F4
                netCam.setDisplayResolution('FHD')
            elif key == 194:  # F5 to toggle fullscreen
                netCam.toggleFullSreen()
        netCam.display()

    netCam.clearAll()
    exit()

