#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

# import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from astropy.io import fits
# matplotlib.use("Qt4Agg")

import random
import pickle
from math import pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QScrollArea,
                            QPushButton, QHBoxLayout, QApplication,
                            QLabel)
from qtpy.QtCore import Qt, QRect, QSize
# from qtpy.QtGui import QKeySequence
# from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import os.path as osp
import argparse


parser = argparse.ArgumentParser(
    description='Transient object labeling')
parser.add_argument('--path',
                    default='/renoir_data_02/jpreyes/stamp_data/filter_r',
                    help="Path to the folder that contains"
                         "transient images")
parser.add_argument('--curves',
                    default='light_curves_25_ids.pickle',
                    help="Path to the file that contains the light curves")


class ImgFit(object):
    def __init__(self, path):
        self.path = path
        self.hdulist = fits.open(path)

        self.ra = float(self.hdulist[0].header["RA_DEG"])
        self.dec = float(self.hdulist[0].header["DEC_DEG"])

        self.data = self.hdulist[1].data
        self.hdulist.close()

    def draw(self, axe, title=""):
        max_val = self.data.mean() + self.data.std()
        min_val = self.data.mean() - self.data.std()
        axe.imshow(self.data, cmap='gray', vmin=min_val,
                   vmax=max_val, interpolation='none')

        if len(title) < 2:
            axe.set_title(self.path[len(self.path) - 10:])
        else:
            axe.set_title(title)


class CL(object):
    def __init__(self, path):
        self.path = path
        self.x = []
        self.y = []
        for i in range(0, 10):
            self.x.append(i)
            self.y.append(random.random())

    def draw(self, axe):
        axe.scatter(self.x, self.y)


class LightCurves(object):
    def __init__(self, path):
        self.path = path
        tmp_pickle = pickle.load(open(self.path, "rb"), encoding='latin1')
        self.ra = []
        self.dec = []
        self.mag = []
        self.times = []
        self.id_names = []
        self.id_times = []
        for obj in tmp_pickle:
            self.ra.append(obj['ra'] * 180. / pi)
            self.dec.append(obj['dec'] * 180. / pi)
            self.mag.append(obj['flux'])
            self.times.append(obj['mjd'])

            idd = obj['id']
            id_tmp_name = []
            id_tmp_times = []
            for i in range(0, len(idd)):
                # print(idd[i])
                a, b = idd[i].decode('utf-8').split("-")
                id_tmp_name.append(int(str(a)))
                id_tmp_times.append(int(b))

            self.id_names.append(id_tmp_name)
            self.id_times.append(id_tmp_times)

    def find_id(self, ra, dec):
        eps = 0.1
        besti, bestt = 0, 0
        for i in range(0, len(self.ra)):
            for t in range(0, len(self.ra[i])):
                abs_1 = abs(self.ra[i][t] - ra)**2
                abs_2 = abs(self.dec[i][t] - dec)
                if abs_1 + abs_2 < eps:
                    eps = abs_1 + abs_2**2
                    besti = i
                    bestt = t
        prec = "Precision: {0} {1} {2} {3} {4} {5} {6}"
        print(prec.format(eps, besti, bestt, ra, dec,
                          self.ra[besti][bestt],
                          self.dec[besti][bestt]))
        return besti, bestt

    def get_value(self, id_, times_):
        return (self.id_names[id_][times_], self.id_times[id_][times_],
                self.ra[id_][times_], self.dec[id_][times_],
                self.mag[id_][times_], self.times[id_][times_])

    def draw(self, axe, obj_it, titre=""):
        axe.scatter(self.times[obj_it], self.mag[obj_it])
        if len(titre) < 2:
            title = 'Obj num: {0} - id: {1}'.format(obj_it,
                                                    self.id_names[obj_it][0])
            axe.set_title(title)
        else:
            axe.set_title(titre)

    def num_objects(self):
        return len(self.ra)

    def object_info(self, num_object):
        return self.id_names[num_object][0], self.id_times[num_object]


class OutputFile:
    def __init__(self, path):
        self.path = path
        self.open()

    def open(self):
        self.name_list = []
        self.type_list = []
        if osp.exists(self.path):
            f = open(self.path, 'r')
            self.read(f)
            f.close()

    def save(self):
        f = open(self.path, 'w')
        for i in range(0, len(self.name_list)):
            line = '{0} {1}\n'.format(self.name_list[i], self.type_list[i])
            f.write(line)
        f.close()

    def read(self, fd):
        file_content = fd.readlines()
        for line in file_content:
            line = line.rstrip()

            self.name_list.append(line.split(" ")[0])
            self.type_list.append(line.split(" ")[1])
            print(line)

    def add_object(self, name, type_):
        if name in self.name_list:
            err = "Can't add object {0}, it was added previously"
            print(err.format(name), file=sys.stderr)
            exit(1)
        else:
            self.name_list.append(name)
            self.type_list.append(type_)

    def change_object(self, name, type_):
        if name in self.name_list:
            n = self.name_list.index(name)
            self.type_list[n] = type_
        else:
            self.name_list.append(name)
            self.type_list.append(type_)

    def find_object(self, name):
        if name in self.name_list:
            return self.type_list[self.name_list.index(name)]
        else:
            return -1


class MatplotlibWidget(FigureCanvas):
    def __init__(self, width, height, parent=None):
        self.fig = Figure()  # figsize=(width, height)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.setGeometry(QRect(0, 0, width, height))
        self.setMinimumSize(QSize(width, height))
        self.setMaximumSize(QSize(width, height))
        self.setAttribute(Qt.WA_Hover)

    def enterEvent(event):
        print("I'm in!")
        FigureCanvas.enterEvent(self, event)

    def leaveEvent(event):
        print("I'm out!")
        FigureCanvas.leaveEvent(self, event)

class MainWindow(QWidget):
    IMAGE_SIZE = 180

    def __init__(self, light_curves, it_object, doc_label,
                 cal_path, diff_path, time_interval):
        super(MainWindow, self).__init__(None)
        self.doc_label = doc_label
        self.light_curves = light_curves
        self.it_object = it_object
        self.cal_path = cal_path
        self.diff_path = diff_path
        self.time_interval = time_interval
        if hasattr(self, 'bigLayout'):
            self.clearLayout(self.bigLayout)
        self.setUpWindows()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_A:
            self.passeImage()
        if key == Qt.Key_Q:
            self.it_object = self.it_object - 2
            self.passeImage()
        if key == Qt.Key_S:
            self.doc_label.save()

        label = -1
        if key == Qt.Key_W:
            label = 0
        if key == Qt.Key_X:
            label = 1
        if key == Qt.Key_C:
            label = 2
        if label != -1:
            object_id, time_list = self.light_curves.object_info(
                self.it_object)
            _id = '{0}_{1}'.format(object_id, self.it_object)
            self.doc_label.add_object(_id, str(label))
            self.doc_label.save()
            self.add_button.setText("A, Q, S :::: Label=" + str(label))
            self.passeImage()

        label = -1
        if key == Qt.Key_K:
            label = 0
        if key == Qt.Key_L:
            label = 1
        if key == Qt.Key_M:
            label = 2
        if label != -1:
            object_id, time_list = self.light_curves.object_info(
                self.it_object)
            _id = '{0}_{1}'.format(object_id, self.it_object)
            self.doc_label.change_object(_id, str(label))
            self.doc_label.save()
            self.add_button.setText("A, Q, S :::: Label=" + str(label))
            self.passeImage()

    def passeImage(self):
        self.it_object = self.it_object + 1
        self.setUpWindows()

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.clearLayout(child.layout())

    def setUpWindows(self):
        object_id, time_list = self.light_curves.object_info(self.it_object)
        _id = '{0}_{1}'.format(object_id, self.it_object)
        current_label = self.doc_label.find_object(_id)

        position = 1
        # espaceJourMinRequis = 1.0
        prev_time = time_list[0]
        line_num = 1
        process_list = [0]
        for i in range(0, len(time_list)):
            if prev_time + self.time_interval < time_list[i]:
                line_num += 1
                prev_time = time_list[i]
                process_list.append(i)

        if hasattr(self, 'bigLayout'):
            self.clearLayout(self.bigLayout)
        else:
            self.bigLayout = QVBoxLayout(self)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        rect = QRect(0, 0, 800, line_num * self.IMAGE_SIZE)
        self.scrollAreaWidgetContents.setGeometry(rect)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.add_button = QPushButton("A, Q, S :::: Label=" + str(current_label))

        self.bigLayout.addWidget(self.add_button)
        self.bigLayout.addWidget(self.scrollArea)

        self.layoutVertical = QHBoxLayout(self.scrollAreaWidgetContents)

        self.layoutGauche = QVBoxLayout(None)
        self.layoutDroit = QVBoxLayout(None)
        self.layoutVertical.addLayout(self.layoutGauche)
        self.layoutVertical.addLayout(self.layoutDroit)

        err = 0
        for i in process_list:
            nom = '{0}-{1}.fits'.format(object_id, i)
            prefix_cal_path = self.cal_path + nom
            prefix_diff_path = self.diff_path + nom

            print(str(object_id), "  :::: ", prefix_cal_path)
            layout_tmp = QHBoxLayout(None)

            path_1 = osp.exists(prefix_cal_path)
            path_2 = osp.exists(prefix_diff_path)
            if not (path_1 or path_2):
                err_msg = "File could not be found {0}".format(nom)
                err = 1
                print(err_msg, file=sys.stderr)

            if err == 0:
                widget_cal_tmp = MatplotlibWidget(self.IMAGE_SIZE,
                                                  self.IMAGE_SIZE, None)
                axe = widget_cal_tmp.axes
                img1 = ImgFit(prefix_cal_path)
                object_curves = self.light_curves.times[self.it_object]
                light_curve = object_curves[time_list[i]]
                img1.draw(axe, "Cal t = {0}".format(light_curve))

                widget_dif_tmp = MatplotlibWidget(self.IMAGE_SIZE,
                                                  self.IMAGE_SIZE, None)
                axe = widget_dif_tmp.axes
                img2 = ImgFit(prefix_diff_path)
                img2.draw(axe, "Diff t = {0}".format(light_curve))
                position = position + 2

                layout_tmp.addWidget(widget_cal_tmp)
                layout_tmp.addWidget(widget_dif_tmp)
                self.layoutGauche.addLayout(layout_tmp)

        widget_CL_tmp = MatplotlibWidget(400, 400, None)
        axe = widget_CL_tmp.axes
        self.light_curves.draw(axe, self.it_object)
        self.layoutDroit.addWidget(widget_CL_tmp)
        self.layoutDroit.addWidget(QLabel(""))

        if err == 1:
            self.passeImage()


if __name__ == '__main__':
    args = parser.parse_args()
    app = QApplication(sys.argv)
    doc_label = OutputFile("label_sortie")

    print("A : ", " Next image")
    print("Q : ", " Previous image")
    print("S : ", " Save")
    print("W : ", " Label 0 (If it doesn't exists)")
    print("X : ", " Label 1 (If it doesn't exists)")
    print("C : ", " Label 2 (If it doesn't exists)")
    print("K : ", " Force label 0 ")
    print("L : ", " Force label 1 ")
    print("M : ", " Force label 2 ")

    # light_curves =
    light_curves = LightCurves(args.curves)
    img_path = args.path
    cal_path = osp.join(img_path, 'cal-')
    diff_path = osp.join(img_path, 'diff-')
    # pathVersCal = "/renoir_data_02/jpreyes/stamp_data/filter_r/cal-"
    # pathVersDiff = "/renoir_data_02/jpreyes/stamp_data/filter_r/diff-"
    time_interval = 7

    main = MainWindow(light_curves, 0, doc_label, cal_path,
                      diff_path, time_interval)
    main.resize(800, 700)
    main.show()
    app.exec_()