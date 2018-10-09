import sys
import cv2
import logging
import configparser

import ultrasound_tracking as ust

# configure logging
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s', level=logging.INFO)


def catch_errors():
    cv2.destroyAllWindows()
    input("Press Enter to continue...")
    sys.exit()

# read config
try:
    configfile = configparser.ConfigParser(allow_no_value=True)
    configfile.read('tracking_config.txt')

    config = {}
    for keys in configfile['General']:
        config[keys] = configfile['General'][keys]

    config['variables'] = {}
    for keys in configfile['Variables']:
        config['variables'][keys] = configfile['Variables'][keys]
except FileNotFoundError as err:
    logging.error('No config file found. ' + format(err))
    catch_errors()
except Exception as err:
    logging.error("Unexpected error: " + format(err))
    catch_errors()



try:
    img = ust.UltrasoundTracking(config)
    # init windows and callback function
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('status', cv2.WINDOW_NORMAL)
    cv2.namedWindow('help', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', img.draw_shape)
except Exception as err:
    logging.error("Unexpected error: " + format(err))
    catch_errors()



# main program loop
while 1:
    try:
        if(img.cvobj is not None):
            cv2.imshow('image', img.cvobj)
        cv2.imshow('status', img.status)
        if img.helpflag:
            cv2.imshow('help', img.helpscreen)
        else:
            cv2.destroyWindow('help')

        k = cv2.waitKey(1) & 0xFF
        # print(k)
        if k == 27:
            break
        elif k == ord('w'):
            img.prev_folder()
        elif k == ord('a'):
            img.prev_image()
        elif k == ord('s'):
            img.next_folder()
        elif k == ord('d'):
            img.next_image()
        elif k == ord('e'):
            img.manual = not img.manual
            img.reload_status()
        elif k == ord('q'):
            img.reset_trial()
        elif k == 32:
            img.mode = not img.mode
            img.reload_status()
        elif k == 9:
            img.next_trial()
        elif k == ord('o'):
            img.export_data()
        elif k in range(49, 56):
            img.tracker_type = ust.TRACKER_TYPES[k - 49]
            img.reload_status()
        elif k == ord('h'):
            img.helpflag = not img.helpflag
    except Exception as err:
        logging.error("Unexpected error: " + format(err))
        catch_errors()

cv2.destroyAllWindows()
