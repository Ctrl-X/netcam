## Main.py is only use to run the NetCam class, nothing particulery interesting inside...

from NetCam import *

if __name__ == '__main__':
    print('Started main.py...')
    netCam = NetCam(display='QVGA', capture='VGA')
    print(netCam.getDetail())
    netCam.startClient()

    try:
        while netCam.isRunning:
            netCam.display()
            key = cv2.waitKey(1)
            if key != -1:
                if key == ord('q'):  # q to quit
                    netCam.clearAll()
                elif key == 35:  # Tilde to show debug
                    netCam.toggleDebug()
                elif key == 190:  # F1
                    netCam.setDisplayResolution('QVGA')
                elif key == 191:  # F2
                    netCam.setDisplayResolution('VGA')
                elif key == 192:  # F3
                    netCam.setDisplayResolution('HD')
                elif key == 193:  # F4
                    netCam.setDisplayResolution('FHD')
                elif key == 194:  # F5
                    netCam.setDisplayResolution('2K')
                elif key == ord('f'):  # F to toggle fullscreen
                    netCam.toggleFullScreen()
                elif key == 27:  # Esc key was pressed,
                    netCam.toggleFullScreen(False)
                else:
                    print(f'Key pressed: {key}')
    except KeyboardInterrupt:
        netCam.clearAll()
    exit()
