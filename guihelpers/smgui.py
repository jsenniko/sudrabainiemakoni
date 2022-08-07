# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'smgui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1146, 872)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.MplWidget1 = MplWidget(self.splitter)
        self.MplWidget1.setEnabled(True)
        self.MplWidget1.setMinimumSize(QtCore.QSize(0, 600))
        self.MplWidget1.setObjectName("MplWidget1")
        self.console = QtWidgets.QPlainTextEdit(self.splitter)
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(11)
        self.console.setFont(font)
        self.console.setObjectName("console")
        self.verticalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1146, 26))
        self.menubar.setObjectName("menubar")
        self.menuFails = QtWidgets.QMenu(self.menubar)
        self.menuFails.setObjectName("menuFails")
        self.menuDarb_bas = QtWidgets.QMenu(self.menubar)
        self.menuDarb_bas.setObjectName("menuDarb_bas")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionIelas_t_att_lu = QtWidgets.QAction(MainWindow)
        self.actionIelas_t_att_lu.setObjectName("actionIelas_t_att_lu")
        self.actionIelas_t_projektu = QtWidgets.QAction(MainWindow)
        self.actionIelas_t_projektu.setObjectName("actionIelas_t_projektu")
        self.actionKalibr_t_kameru = QtWidgets.QAction(MainWindow)
        self.actionKalibr_t_kameru.setObjectName("actionKalibr_t_kameru")
        self.actionSaglab_t_projektu = QtWidgets.QAction(MainWindow)
        self.actionSaglab_t_projektu.setObjectName("actionSaglab_t_projektu")
        self.actionHorizont_lo_koordin_tu_re_is = QtWidgets.QAction(MainWindow)
        self.actionHorizont_lo_koordin_tu_re_is.setObjectName("actionHorizont_lo_koordin_tu_re_is")
        self.menuFails.addAction(self.actionIelas_t_att_lu)
        self.menuFails.addAction(self.actionKalibr_t_kameru)
        self.menuFails.addSeparator()
        self.menuFails.addAction(self.actionIelas_t_projektu)
        self.menuFails.addAction(self.actionSaglab_t_projektu)
        self.menuDarb_bas.addAction(self.actionHorizont_lo_koordin_tu_re_is)
        self.menubar.addAction(self.menuFails.menuAction())
        self.menubar.addAction(self.menuDarb_bas.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFails.setTitle(_translate("MainWindow", "Fails"))
        self.menuDarb_bas.setTitle(_translate("MainWindow", "Darbības"))
        self.actionIelas_t_att_lu.setText(_translate("MainWindow", "Ielasīt attēlu"))
        self.actionIelas_t_projektu.setText(_translate("MainWindow", "Ielasīt projektu"))
        self.actionKalibr_t_kameru.setText(_translate("MainWindow", "Kalibrēt kameru"))
        self.actionSaglab_t_projektu.setText(_translate("MainWindow", "Saglabāt projektu"))
        self.actionHorizont_lo_koordin_tu_re_is.setText(_translate("MainWindow", "Horizontālo koordinātu režģis"))

from mplwidget import MplWidget
