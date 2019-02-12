import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import simplejson
import datetime, time
import logging

# global constants

WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
TRACKER_TYPES = ('BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT')


# main image tracking class
class UltrasoundTracking:
    ntrials = 5

    def __init__(self, config):
        self.config = config
        self.parse_walkdir()
        self.drawing = False  # true if mouse is pressed
        self.mode = True  # if True, draw rectangle for object, else for fixpoint
        self.manual = False  # if True, draw rectangles manual without tracking
        self.helpflag = False  # if True, show helpscreen
        self.ix, self.iy = -1, -1
        self.x, self.y = 10, 10
        self.kfold = 0
        self.kfile = 0
        self.trial = 0
        self.tracker_type = TRACKER_TYPES[1]
        self.nfolder = self.folderlist.__len__()
        self.reload_folder()
        self.read_tracking()
        self.reload_image()
        self.plot_distance()
        self.helpscreen = self.draw_helpscreen()

        # init plotting
        plt.style.use('seaborn-whitegrid')
        plt.ion()
        plt.ylim((-10, 10))
        plt.xlim((-10, 10))
        plt.show()

    def parse_walkdir(self):
        # set walking directory
        self.walk_dir = os.path.abspath(self.config['folder'])
        self.folderlist = []
        for root, subdirs, files in os.walk(self.walk_dir):
            if not subdirs:
                self.folderlist.append(root)

    def reload_status(self):
        self.status = np.zeros((512, 700, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        nervepoint = self.get_point('nerve')
        fixpoint = self.get_point('fix')

        cv2.putText(self.status, 'Folder:', (10, 30), font, 1, WHITE, 2, cv2.LINE_AA)
        cv2.putText(self.status, '{0} / {1}'.format(self.kfold + 1, self.nfolder), (150, 30), font, 1, WHITE,
                    2, cv2.LINE_AA)
        cv2.putText(self.status, self.folder.replace(self.walk_dir, ''), (300, 30), font, .5, WHITE, 1, cv2.LINE_AA)

        cv2.putText(self.status, 'Trial:', (10, 80), font, 1, WHITE, 2, cv2.LINE_AA)
        cv2.putText(self.status, '{0} / {1}'.format(self.trial + 1, self.ntrials), (150, 80), font, 1, WHITE,
                    2, cv2.LINE_AA)

        cv2.putText(self.status, 'objX:', (10, 200), font, 1, WHITE, 2, cv2.LINE_AA)
        cv2.putText(self.status, 'objY:', (10, 250), font, 1, WHITE, 2, cv2.LINE_AA)

        cv2.putText(self.status, 'fixX:', (10, 320), font, 1, WHITE, 2, cv2.LINE_AA)
        cv2.putText(self.status, 'fixY:', (10, 370), font, 1, WHITE, 2, cv2.LINE_AA)

        cv2.putText(self.status, 'Tracker:', (10, 440), font, 1, WHITE, 2, cv2.LINE_AA)
        cv2.putText(self.status, self.tracker_type, (150, 440), font, 1, WHITE, 2, cv2.LINE_AA)

        cv2.putText(self.status, "Press 'h' to toggle help", (10, 500), font, .5, WHITE, 1, cv2.LINE_AA)

        if nervepoint:
            x, y = nervepoint.values()
            cv2.putText(self.status, str(x), (150, 200), font, 1, WHITE, 2, cv2.LINE_AA)
            cv2.putText(self.status, str(y), (150, 250), font, 1, WHITE, 2, cv2.LINE_AA)

        if fixpoint:
            x, y = fixpoint.values()
            cv2.putText(self.status, str(x), (150, 320), font, 1, WHITE, 2, cv2.LINE_AA)
            cv2.putText(self.status, str(y), (150, 370), font, 1, WHITE, 2, cv2.LINE_AA)

        if self.imglist:
            status = self.get_trial_status(self.trial)

            imgstatusheight = 400
            n_minus = max(self.n -1, 1)
            rectspace = int((0.1 * imgstatusheight) / (n_minus))
            space = 20

            rectheight = int((imgstatusheight - (n_minus) * rectspace) / self.n)
            rectwidth = 25

            rectsize = 25

            xstart = 400
            ystart = 80
            for k in range(self.n):
                color = GREEN if status['nerve'][k] else RED
                cv2.rectangle(self.status, (xstart, ystart + (rectspace + rectheight) * k),
                              (xstart + rectwidth, ystart + rectheight + (rectspace + rectheight) * k), color, -1)
                color = GREEN if status['fix'][k] else RED
                cv2.rectangle(self.status, (xstart + rectwidth + space, ystart + (rectspace + rectheight) * k),
                              (xstart + space + 2 * rectwidth, ystart + rectheight + (rectspace + rectheight) * k),
                              color,
                              -1)

            if self.manual:
                cv2.rectangle(self.status, (10, 400), (300, 500), YELLOW, -1)
                cv2.putText(self.status, 'MANUAL', (40, 470), font, 2, (0, 0, 0), 2, cv2.LINE_AA)

            if self.mode:
                cv2.rectangle(self.status, (xstart - 5, ystart - 5),
                              (xstart + rectwidth + 5,
                               ystart + rectheight + (rectspace + rectheight) * (self.n - 1) + 5),
                              YELLOW, 2)
            else:
                cv2.rectangle(self.status, (xstart + rectsize + space - 5, ystart - 5),
                              (xstart + space + 2 * rectwidth + 5,
                               ystart + rectheight + (rectspace + rectheight) * (self.n - 1) + 5), YELLOW, 2)

            # test = self.gather_points()
            self.draw_arrow(xstart - 40, ystart + int(rectheight / 2) + self.kfile * (rectheight + rectspace))

    def draw_arrow(self, x, y):
        length = 20
        cv2.arrowedLine(self.status, (x, y), (x + length, y), WHITE, 2, tipLength=.5)

    def gather_points(self):
        dflist = []
        for trial in range(self.ntrials):
            for object in ['nerve', 'fix']:
                for k in range(self.n):
                    tstamp = self.tracking[trial][k][object]['time']
                    points = dict(self.tracking[trial][k][object]['point'])
                    if points:
                        points.update(self.parse_folder_to_vars())
                        points['folder'] = self.folder
                        points['trial'] = trial + 1
                        points['time'] = tstamp
                        points['file'] = self.imglist[k]
                        points['image_no'] = k + 1
                        points['object'] = object
                        dflist.append(points)
                    # self.tracking[self.trial][k]['fix']['point']
        df = pd.DataFrame(dflist)
        return df

    def export_data(self):
        temp = []
        for k in range(self.nfolder):
            temp.append(self.gather_points())
            self.next_folder()

        out_df = pd.concat(temp)
        out_df[['x', 'y']] = out_df[['x', 'y']] * float(self.config['scaling_factor'])
        try:
            out_df.to_csv(os.path.join(self.config['folder'], 'results.csv'), sep=";", decimal=",", index=False)
        except PermissionError as err:
            logging.error('Cannot access output file: ' + format(err))
        else:
            logging.info('Output successfully written.')

    def parse_folder_to_vars(self):
        vars = self.config['variables'].values()
        varvals = self.folder.split('\\')[-len(vars):]
        var_dict = {key: value for key, value in zip(vars, varvals)}
        return var_dict

    def plot_distance(self):
        if self.imglist and self.n > 1:
            plotted = False
            df = self.gather_points()
            plt.clf()
            pointsize = np.ones(self.n - 1)
            pointsize[len(pointsize) - 1] = 5

            for trial in range(self.ntrials):
                status = self.get_trial_status(trial)
                if all(status['nerve']) & all(status['fix']):
                    plotted = True
                    xdiff = np.cumsum(
                        np.diff(df[(df['object'] == 'nerve') & (df['trial'] == trial + 1)]['x'])) - np.cumsum(np.diff(
                        df[(df['object'] == 'fix') & (df['trial'] == trial + 1)]['x']))
                    ydiff = np.cumsum(
                        np.diff(df[(df['object'] == 'nerve') & (df['trial'] == trial + 1)]['y'])) - np.cumsum(np.diff(
                        df[(df['object'] == 'fix') & (df['trial'] == trial + 1)]['y']))
                    plt.plot(xdiff, ydiff, label=trial + 1)
                    plt.scatter(xdiff, ydiff, s=pointsize * 10)
                    # plt.pause(0.001)

            if plotted:
                plt.legend()

    def write_tracking(self):
        f = open(os.path.join(self.folder, 'tracking.json'), "w")
        try:
            f.write(simplejson.dumps(self.tracking, indent=4))
        finally:
            f.close()

    def read_tracking(self):
        self.reload_folder()
        try:
            f = open(os.path.join(self.folder, 'tracking.json'))
            self.tracking = simplejson.load(f)
            f.close()
        except:
            self.reset_tracking()


        if not self.tracking and self.n > 0:
            self.reset_tracking()

    def get_trial_status(self, trial):
        status = {'nerve': [], 'fix': []}
        for k in range(self.n):
            status['nerve'].append(bool(self.tracking[trial][k]['nerve']['point']))
            status['fix'].append(bool(self.tracking[trial][k]['fix']['point']))
        return status

    def reset_tracking(self):
        self.tracking = []  # trial level

        for trial in range(self.ntrials):
            self.tracking.append([])  # trial level
            self.add_trial(trial)

        self.write_tracking()
        self.plot_distance()
        self.reload_image()

    def reset_trial(self):
        if self.tracking:
            self.tracking[self.trial] = []
            if self.n > 0:
                self.add_trial(self.trial)
            self.write_tracking()
            self.plot_distance()
            self.reload_image()

    def add_trial(self, trial):
        for k in range(self.n):
            self.tracking[trial].append(
                {'nerve': {
                    'rect': {},
                    'point': {},
                    'time': ""
                },
                    'fix': {
                        'rect': {},
                        'point': {},
                        'time': ""
                    }}
            )

    def reload_folder(self):
        if self.folderlist:
            self.folder = self.folderlist[self.kfold]
            self.imglist = []
            for file in os.listdir(self.folder):
                if file.endswith(self.config["image_format"]):
                    self.imglist.append(file)
            self.n = self.imglist.__len__()
        else:
            logging.error('Folder not found.')
            sys.exit()

    def reload_image(self):
        # self.reload_folder()
        if self.imglist:
            self.imgfile = self.imglist[self.kfile]
            self.imgpath = os.path.join(self.folder, self.imgfile)
            self.cvobj = cv2.cvtColor(cv2.imread(self.imgpath, 0), 8)
            # self.show_ellipse()
            self.show_rectangle('nerve')
            self.show_rectangle('fix')
            self.reload_status()
            self.plot_distance()
        else:
            self.cvobj = np.zeros((512, 512, 3), np.uint8)
            self.tracking = []
            self.reload_status()
            logging.error('No images to display with format ' + self.config["image_format"])

    def next_trial(self):
        global mode
        self.trial += 1
        if self.trial == self.ntrials:
            self.trial = 0
        # self.read_tracking()
        mode = True
        self.reload_image()

    def next_image(self):
        self.kfile += 1
        if self.kfile >= self.n:
            self.kfile = 0
        self.reload_image()

    def prev_image(self):
        self.kfile -= 1
        if self.kfile <= -1:
            self.kfile = self.n - 1
        self.reload_image()

    def next_folder(self):
        global mode
        mode = True
        self.write_tracking()
        self.kfold += 1
        self.kfile = 0
        if self.kfold == self.nfolder:
            self.kfold = 0
        self.read_tracking()
        self.reload_image()

    def prev_folder(self):
        global mode
        mode = True
        self.write_tracking()
        self.kfold -= 1
        self.kfile = 0
        if self.kfold == -1:
            self.kfold = self.nfolder - 1
        self.read_tracking()
        self.reload_image()

    def set_rectangle(self, rect, target):
        x1, y1, x2, y2 = rect.values()
        self.tracking[self.trial][self.kfile][target]['rect'] = rect

        point = {'x': int((x1 + x2) / 2),
                 'y': int((y1 + y2) / 2)}
        self.tracking[self.trial][self.kfile][target]['point'] = point

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.tracking[self.trial][self.kfile][target]['time'] = timestamp

    def remove_rectangle(self, target):
        self.tracking[self.trial][self.kfile][target]['rect'] = {}
        self.tracking[self.trial][self.kfile][target]['point'] = {}
        self.tracking[self.trial][self.kfile][target]['timestamp'] = ""
        self.write_tracking()

    def get_rectangle(self, target):
        if self.tracking:
            return self.tracking[self.trial][self.kfile][target]['rect']
        else:
            return {}

    def get_point(self, target):
        if self.tracking:
            return self.tracking[self.trial][self.kfile][target]['point']
        else:
            return {}

    def show_rectangle(self, target):
        if target == 'nerve':
            color = GREEN
        else:
            color = YELLOW

        rect = self.get_rectangle(target)
        if rect:
            x1, y1, x2, y2 = rect.values()
            cv2.rectangle(self.cvobj,
                          (x1, y1),
                          (x2, y2),
                          color, 1)
            cv2.line(self.cvobj,
                     (int((x1 + x2) / 2), y1),
                     (int((x1 + x2) / 2), y2),
                     RED, 1)
            cv2.line(self.cvobj,
                     (x1, int((y1 + y2) / 2)),
                     (x2, int((y1 + y2) / 2)),
                     RED, 1)

    def track_rectangle(self, target):

        if self.tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if self.tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if self.tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if self.tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if self.tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if self.tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if self.tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = cv2.TrackerMIL_create()

        rect = self.get_rectangle(target)
        if rect:
            x1, y1, x2, y2 = rect.values()
            bbox = (x1, y1, x2 - x1, y2 - y1)

            if bbox[2] < 0:
                bbox = (bbox[0] + bbox[2], bbox[1], -bbox[2], bbox[3])

            if bbox[3] < 0:
                bbox = (bbox[0], bbox[1] + bbox[3], bbox[2], -bbox[3])

            tracker.init(self.cvobj, bbox)

            for k in range(self.n - 1):
                self.next_image()
                ok, bbox = tracker.update(self.cvobj)

                # Draw bounding box
                if ok:
                    # Tracking success
                    rect = {'x1': int(bbox[0]), 'y1': int(bbox[1]),
                            'x2': int(bbox[0] + bbox[2]),
                            'y2': int(bbox[1] + bbox[3])}
                    self.set_rectangle(rect, target)
                    logging.info('Successful tracking on image ' + self.imgfile + ' with tracker ' + self.tracker_type)
                else:
                    # Tracking failure
                    self.remove_rectangle(target)
                    logging.error('Tracking error on image ' + self.imgfile + ' with tracker ' + self.tracker_type)
            self.next_image()
        self.write_tracking()

    # mouse callback function
    def draw_shape(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            #time.sleep(100)
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            rect = {'x1': self.ix, 'y1': self.iy,
                    'x2': x, 'y2': y}
            if self.drawing:
                if self.mode:
                    self.set_rectangle(rect, 'nerve')
                else:
                    self.set_rectangle(rect, 'fix')

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if not self.manual:
                if self.mode:
                    self.track_rectangle('nerve')
                else:
                    self.track_rectangle('fix')
            self.write_tracking()
            self.plot_distance()

        self.reload_image()

    # dummy function
    def nothing(x):
        pass

    # make helpscreen
    def draw_helpscreen(self):
        helpscreen = np.zeros((512, 700, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(helpscreen, "Navigation:", (10, 30), font, .5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(helpscreen, "W:   previous folder", (10, 50), font, .4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(helpscreen, "A:   previous image", (10, 70), font, .4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(helpscreen, "S:   next folder", (10, 90), font, .4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(helpscreen, "D:   next image", (10, 110), font, .4, WHITE, 1, cv2.LINE_AA)

        cv2.putText(helpscreen, "Selection:", (10, 140), font, .5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(helpscreen, "TAB:   skip through trials", (10, 160), font, .4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(helpscreen, "SPACE:   change selection target (object or fixpoint)", (10, 180), font, .4, WHITE, 1,
                    cv2.LINE_AA)
        cv2.putText(helpscreen, "E:   change selection mode (manual or tracking)", (10, 200), font, .4, WHITE, 1,
                    cv2.LINE_AA)
        cv2.putText(helpscreen, "1-7:   Select different tracking algorithms", (10, 220), font, .4, WHITE, 1,
                    cv2.LINE_AA)

        cv2.putText(helpscreen, "Manipulation:", (10, 250), font, .5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(helpscreen, "Q:   Reset rectangles for current folder", (10, 270), font, .4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(helpscreen, "O:   Create output csv file in working folder", (10, 290), font, .4, WHITE, 1,
                    cv2.LINE_AA)

        cv2.putText(helpscreen, "WARNING: All changes to rectangles are saved immediately to tracking.json files",
                    (10, 350), font, .5, YELLOW, 1,
                    cv2.LINE_AA)

        return helpscreen

