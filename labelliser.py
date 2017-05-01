#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from astropy.io import fits
#matplotlib.use("Qt4Agg")

import random
import pickle
import astropy
from astropy.coordinates import ICRS, Galactic
from astropy import units as u
from astropy.coordinates import SkyCoord
from math import pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QScrollArea,
                            QPushButton, QHBoxLayout, QApplication,
                            QLabel)
from qtpy.QtCore import Qt, Signal, QRect, QSize
# from qtpy.QtGui import QKeySequence
# from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import os.path as osp


class ImgFit(object):
    def __init__(self, chemin):
        self.chemin = chemin
        self.hdulist = fits.open(chemin)

        self.ra = float(self.hdulist[0].header["RA_DEG"])
        self.dec = float(self.hdulist[0].header["DEC_DEG"])

        self.donnees = self.hdulist[1].data
        self.hdulist.close()

    def draw(self, axe, titre=""):
        max_val = self.donnees.mean() + self.donnees.std()
        min_val = self.donnees.mean() - self.donnees.std()
        axe.imshow(self.donnees, cmap='gray', vmin=min_val,
                   vmax=max_val, interpolation='none')

        if len(titre) < 2:
            axe.set_title(self.chemin[len(self.chemin) - 10:])
        else:
            axe.set_title(titre)


class CL(object):
    def __init__(self, chemin):
        self.chemin = chemin
        self.x = []
        self.y = []
        for i in range(0, 10):
            self.x.append(i)
            self.y.append(random.random())

    def draw(self, axe):
        axe.scatter(self.x, self.y)


class CourbesLumiere:
    def __init__(self, chemin):
        self.chemin = chemin
        tmp_pickle = pickle.load(open(self.chemin, "rb"), encoding='latin1')
        self.ra = []
        self.dec = []
        self.mag = []
        self.temps = []
        self.id_nom = []
        self.id_temps = []
        for objet in tmp_pickle:
            self.ra.append(objet['ra'] * 180. / pi)
            self.dec.append(objet['dec'] * 180. / pi)
            self.mag.append(objet['flux'])
            self.temps.append(objet['mjd'])

            idd = objet['id']
            id_tmp_nom = []
            id_tmp_temps = []
            for i in range(0, len(idd)):
                # print(idd[i])
                a, b = idd[i].decode('utf-8').split("-")
                id_tmp_nom.append(int(str(a)))
                id_tmp_temps.append(int(b))

            self.id_nom.append(id_tmp_nom)
            self.id_temps.append(id_tmp_temps)

    def rechercheID(self, ra, dec):
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

    def getValue(self, id_, temps_):
        return (self.id_nom[id_][temps_], self.id_temps[id_][temps_],
                self.ra[id_][temps_], self.dec[id_][temps_],
                self.mag[id_][temps_], self.temps[id_][temps_])

    def draw(self, axe, objet_it, titre=""):
        axe.scatter(self.temps[objet_it], self.mag[objet_it])
        if len(titre) < 2:
            title = 'Obj num: {0} - id: {1}'.format(objet_it,
                                                    self.id_nom[objet_it][0])
            axe.set_title(title)
        else:
            axe.set_title(titre)

    def nbObjets(self):
        return len(self.ra)

    def infoObjet(self, objet_num):
        return self.id_nom[objet_num][0], self.id_temps[objet_num]


class DocumentSortie:
    def __init__(self, chemin):
        self.chemin = chemin
        self.lecture()

    def lecture(self):
        self.liste_nom = []
        self.liste_type = []
        if osp.exists(self.chemin):
            f = open(self.chemin, 'r')
            self.lire(f)
            f.close()

    def save(self):
        f = open(self.chemin, 'w')
        for i in range(0, len(self.liste_nom)):
            line = '{0} {1}\n'.format(self.liste_nom[i], self.liste_type[i])
            f.write(line)
        f.close()

    def lire(self, fd):
        file_content = fd.readlines()
        for line in file_content:
            line = line.rstrip()

            self.liste_nom.append(line.split(" ")[0])
            self.liste_type.append(line.split(" ")[1])
            print(line)

    def addObjet(self, nom_, type_):
        if(nom_ in self.liste_nom):
            print("AJOUT IMPOSSIBLE ", nom_, " DEJA EXISTANT")
            exit(1)
        else:
            self.liste_nom.append(nom_)
            self.liste_type.append(type_)

    def changerObjet(self, nom_, type_):
        if nom_ in self.liste_nom:
            n = self.liste_nom.index(nom_)
            self.liste_type[n] = type_
        else:
            self.liste_nom.append(nom_)
            self.liste_type.append(type_)

    def trouverObjet(self, nom_):
        if nom_ in self.liste_nom:
            return self.liste_type[self.liste_nom.index(nom_)]
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


class Fenetre(QWidget):
    TAILLE_IMAGE = 180

    def __init__(self, coubresLumiere, it_objet, docLabel):
        super(Fenetre, self).__init__(None)
        self.docLabel = docLabel
        self.coubresLumiere = coubresLumiere
        self.it_objet = it_objet
        if hasattr(self, 'bigLayout'):
            self.clearLayout(self.bigLayout)
        self.setUpWindows()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_A:
            self.passeImage()
        if key == Qt.Key_Q:
            self.it_objet = self.it_objet - 2
            self.passeImage()
        if key == Qt.Key_S:
            docLabel.save()

        label = -1
        if key == Qt.Key_W:
            label = 0
        if key == Qt.Key_X:
            label = 1
        if key == Qt.Key_C:
            label = 2
        if label != -1:
            id_objet, liste_temps = self.coubresLumiere.infoObjet(
                self.it_objet)
            _id = '{0}_{1}'.format(id_objet, self.it_objet)
            docLabel.addObjet(_id, str(label))
            docLabel.save()
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
            id_objet, liste_temps = self.coubresLumiere.infoObjet(
                self.it_objet)
            _id = '{0}_{1}'.format(id_objet, self.it_objet)
            docLabel.changerObjet(_id, str(label))
            docLabel.save()
            self.add_button.setText("A, Q, S :::: Label=" + str(label))
            self.passeImage()

    def passeImage(self):
        self.it_objet = self.it_objet + 1
        self.setUpWindows()

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.clearLayout(child.layout())

    def setUpWindows(self):
        id_objet, liste_temps = self.coubresLumiere.infoObjet(self.it_objet)
        _id = '{0}_{1}'.format(id_objet, self.it_objet)
        labelActuel = self.docLabel.trouverObjet(_id)

        position = 1
        # espaceJourMinRequis = 1.0
        dernierTemps = liste_temps[0]
        nbLigne = 1
        listeATraiter = [0]
        for i in range(0, len(liste_temps)):
            if dernierTemps + espaceTemps < liste_temps[i]:
                nbLigne += 1
                dernierTemps = liste_temps[i]
                listeATraiter.append(i)

        if hasattr(self, 'bigLayout'):
            self.clearLayout(self.bigLayout)
        else:
            self.bigLayout = QVBoxLayout(self)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        rect = QRect(0, 0, 800, nbLigne * self.TAILLE_IMAGE)
        self.scrollAreaWidgetContents.setGeometry(rect)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.add_button = QPushButton("A, Q, S :::: Label=" + str(labelActuel))

        self.bigLayout.addWidget(self.add_button)
        self.bigLayout.addWidget(self.scrollArea)

        self.layoutVertical = QHBoxLayout(self.scrollAreaWidgetContents)

        self.layoutGauche = QVBoxLayout(None)
        self.layoutDroit = QVBoxLayout(None)
        self.layoutVertical.addLayout(self.layoutGauche)
        self.layoutVertical.addLayout(self.layoutDroit)

        err = 0
        for i in listeATraiter:
            nom = '{0}-{1}.fits'.format(id_objet, i)
            prefix_cal_path = cheminVersCal + nom
            prefix_diff_path = cheminVersDiff + nom

            print(str(id_objet), "  :::: ", prefix_cal_path)
            layout_tmp = QHBoxLayout(None)

            path_1 = osp.exists(prefix_cal_path)
            path_2 = osp.exists(prefix_diff_path)
            if not (path_1 or path_2):
                err_msg = "File could not be found {0}".format(nom)
                err = 1
                print(err_msg, file=sys.stderr)

            if err == 0:
                widget_cal_tmp = MatplotlibWidget(self.TAILLE_IMAGE,
                                                  self.TAILLE_IMAGE, None)
                axe = widget_cal_tmp.axes
                img1 = ImgFit(prefix_cal_path)
                object_curves = self.coubresLumiere.temps[self.it_objet]
                light_curve = object_curves[liste_temps[i]]
                img1.draw(axe, "Cal t = {0}".format(light_curve))

                widget_dif_tmp = MatplotlibWidget(self.TAILLE_IMAGE,
                                                  self.TAILLE_IMAGE, None)
                axe = widget_dif_tmp.axes
                img2 = ImgFit(prefix_diff_path)
                img2.draw(axe, "Diff t = {0}".format(light_curve))
                position = position + 2

                layout_tmp.addWidget(widget_cal_tmp)
                layout_tmp.addWidget(widget_dif_tmp)
                self.layoutGauche.addLayout(layout_tmp)

        widget_CL_tmp = MatplotlibWidget(400, 400, None)
        axe = widget_CL_tmp.axes
        self.coubresLumiere.draw(axe, self.it_objet)
        self.layoutDroit.addWidget(widget_CL_tmp)
        self.layoutDroit.addWidget(QLabel(""))

        if err == 1:
            self.passeImage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    docLabel = DocumentSortie("label_sortie")

    print("A : ", " img suivante")
    print("Q : ", " img precedente")
    print("S : ", " save")
    print("W : ", " label 0 (si non existant)")
    print("X : ", " label 1 (si non existant)")
    print("C : ", " label 2 (si non existant)")
    print("K : ", " force label 0 ")
    print("L : ", " force label 1 ")
    print("M : ", " force label 2 ")

    coubresLumiere = CourbesLumiere("light_curves_25_ids.pickle")
    cheminVersCal = "/renoir_data_02/jpreyes/stamp_data/filter_r/cal-"
    cheminVersDiff = "/renoir_data_02/jpreyes/stamp_data/filter_r/diff-"
    espaceTemps = 7

    main = Fenetre(coubresLumiere, 0, docLabel)
    main.resize(800, 700)
    main.show()
    app.exec_()
