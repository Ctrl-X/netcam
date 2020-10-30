from NetCam import *

if __name__ == "__main__":
    netCam = NetCam(display='HD', ip='192.168.1.70', port='5556')
    # netCam.invertVertical()

    try:
        while netCam.isRunning():
            netCam.display()
    except KeyboardInterrupt:
        netCam.clearAll()
    except Exception as err:
        netCam.clearAll()
    exit()
