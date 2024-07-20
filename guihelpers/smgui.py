# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'smgui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


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
        self.menuZ_m_t = QtWidgets.QMenu(self.menubar)
        self.menuZ_m_t.setObjectName("menuZ_m_t")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionIelas_t_att_lu = QtWidgets.QAction(MainWindow)
        self.actionIelas_t_att_lu.setObjectName("actionIelas_t_att_lu")
        self.actionIelas_t_projektu = QtWidgets.QAction(MainWindow)
        self.actionIelas_t_projektu.setObjectName("actionIelas_t_projektu")
        self.actionKalibr_t_kameru = QtWidgets.QAction(MainWindow)
        self.actionKalibr_t_kameru.setEnabled(False)
        self.actionKalibr_t_kameru.setObjectName("actionKalibr_t_kameru")
        self.actionSaglab_t_projektu = QtWidgets.QAction(MainWindow)
        self.actionSaglab_t_projektu.setEnabled(False)
        self.actionSaglab_t_projektu.setObjectName("actionSaglab_t_projektu")
        self.actionHorizont_lo_koordin_tu_re_is = QtWidgets.QAction(MainWindow)
        self.actionHorizont_lo_koordin_tu_re_is.setEnabled(False)
        self.actionHorizont_lo_koordin_tu_re_is.setObjectName("actionHorizont_lo_koordin_tu_re_is")
        self.actionCiparot_zvaigznes = QtWidgets.QAction(MainWindow)
        self.actionCiparot_zvaigznes.setEnabled(False)
        self.actionCiparot_zvaigznes.setObjectName("actionCiparot_zvaigznes")
        self.actionAtt_lu = QtWidgets.QAction(MainWindow)
        self.actionAtt_lu.setEnabled(False)
        self.actionAtt_lu.setObjectName("actionAtt_lu")
        self.actionProjic_t = QtWidgets.QAction(MainWindow)
        self.actionProjic_t.setEnabled(False)
        self.actionProjic_t.setObjectName("actionProjic_t")
        self.actionProjekcijas_apgabals = QtWidgets.QAction(MainWindow)
        self.actionProjekcijas_apgabals.setObjectName("actionProjekcijas_apgabals")
        self.actionKartes_apgabals = QtWidgets.QAction(MainWindow)
        self.actionKartes_apgabals.setObjectName("actionKartes_apgabals")
        self.actionIelas_t_otro_projektu = QtWidgets.QAction(MainWindow)
        self.actionIelas_t_otro_projektu.setEnabled(False)
        self.actionIelas_t_otro_projektu.setObjectName("actionIelas_t_otro_projektu")
        self.actionProjic_t_kop = QtWidgets.QAction(MainWindow)
        self.actionProjic_t_kop.setEnabled(False)
        self.actionProjic_t_kop.setObjectName("actionProjic_t_kop")
        self.actionKontrolpunkti = QtWidgets.QAction(MainWindow)
        self.actionKontrolpunkti.setEnabled(False)
        self.actionKontrolpunkti.setObjectName("actionKontrolpunkti")
        self.actionIelas_t_kontrolpunktus = QtWidgets.QAction(MainWindow)
        self.actionIelas_t_kontrolpunktus.setEnabled(False)
        self.actionIelas_t_kontrolpunktus.setObjectName("actionIelas_t_kontrolpunktus")
        self.actionKontrolpunktu_augstumus = QtWidgets.QAction(MainWindow)
        self.actionKontrolpunktu_augstumus.setEnabled(False)
        self.actionKontrolpunktu_augstumus.setObjectName("actionKontrolpunktu_augstumus")
        self.actionIzveidot_augstumu_karti = QtWidgets.QAction(MainWindow)
        self.actionIzveidot_augstumu_karti.setEnabled(False)
        self.actionIzveidot_augstumu_karti.setObjectName("actionIzveidot_augstumu_karti")
        self.actionSaglab_t_augstumu_karti = QtWidgets.QAction(MainWindow)
        self.actionSaglab_t_augstumu_karti.setEnabled(False)
        self.actionSaglab_t_augstumu_karti.setObjectName("actionSaglab_t_augstumu_karti")
        self.actionIelas_t_augstumu_karti = QtWidgets.QAction(MainWindow)
        self.actionIelas_t_augstumu_karti.setEnabled(False)
        self.actionIelas_t_augstumu_karti.setObjectName("actionIelas_t_augstumu_karti")
        self.actionAugstumu_karti = QtWidgets.QAction(MainWindow)
        self.actionAugstumu_karti.setEnabled(False)
        self.actionAugstumu_karti.setObjectName("actionAugstumu_karti")
        self.actionProjic_t_no_augstumu_kartes = QtWidgets.QAction(MainWindow)
        self.actionProjic_t_no_augstumu_kartes.setEnabled(False)
        self.actionProjic_t_no_augstumu_kartes.setObjectName("actionProjic_t_no_augstumu_kartes")
        self.actionProjic_t_kop_no_augstumu_kartes = QtWidgets.QAction(MainWindow)
        self.actionProjic_t_kop_no_augstumu_kartes.setEnabled(False)
        self.actionProjic_t_kop_no_augstumu_kartes.setObjectName("actionProjic_t_kop_no_augstumu_kartes")
        self.actionKameras_kalibr_cijas_parametri = QtWidgets.QAction(MainWindow)
        self.actionKameras_kalibr_cijas_parametri.setObjectName("actionKameras_kalibr_cijas_parametri")
        self.actionSaglab_t_projic_to_att_lu_JPG = QtWidgets.QAction(MainWindow)
        self.actionSaglab_t_projic_to_att_lu_JPG.setEnabled(False)
        self.actionSaglab_t_projic_to_att_lu_JPG.setObjectName("actionSaglab_t_projic_to_att_lu_JPG")
        self.actionSaglab_t_projic_to_att_lu_TIFF = QtWidgets.QAction(MainWindow)
        self.actionSaglab_t_projic_to_att_lu_TIFF.setEnabled(False)
        self.actionSaglab_t_projic_to_att_lu_TIFF.setObjectName("actionSaglab_t_projic_to_att_lu_TIFF")
        self.actionIelas_t_kameru = QtWidgets.QAction(MainWindow)
        self.actionIelas_t_kameru.setEnabled(False)
        self.actionIelas_t_kameru.setObjectName("actionIelas_t_kameru")
        self.actionSaglab_t_kameru = QtWidgets.QAction(MainWindow)
        self.actionSaglab_t_kameru.setEnabled(False)
        self.actionSaglab_t_kameru.setObjectName("actionSaglab_t_kameru")
        self.actionUzst_d_t_datumu = QtWidgets.QAction(MainWindow)
        self.actionUzst_d_t_datumu.setEnabled(True)
        self.actionUzst_d_t_datumu.setObjectName("actionUzst_d_t_datumu")
        self.actionUzst_d_t_platumu_garumu_augstumu = QtWidgets.QAction(MainWindow)
        self.actionUzst_d_t_platumu_garumu_augstumu.setObjectName("actionUzst_d_t_platumu_garumu_augstumu")
        self.menuFails.addAction(self.actionIelas_t_att_lu)
        self.menuFails.addSeparator()
        self.menuFails.addAction(self.actionIelas_t_projektu)
        self.menuFails.addAction(self.actionSaglab_t_projektu)
        self.menuFails.addAction(self.actionIelas_t_otro_projektu)
        self.menuFails.addSeparator()
        self.menuFails.addAction(self.actionIelas_t_kontrolpunktus)
        self.menuFails.addSeparator()
        self.menuFails.addAction(self.actionIelas_t_augstumu_karti)
        self.menuFails.addAction(self.actionSaglab_t_augstumu_karti)
        self.menuFails.addSeparator()
        self.menuFails.addAction(self.actionSaglab_t_projic_to_att_lu_JPG)
        self.menuFails.addAction(self.actionSaglab_t_projic_to_att_lu_TIFF)
        self.menuFails.addSeparator()
        self.menuFails.addAction(self.actionIelas_t_kameru)
        self.menuFails.addAction(self.actionSaglab_t_kameru)
        self.menuDarb_bas.addAction(self.actionAtt_lu)
        self.menuDarb_bas.addSeparator()
        self.menuDarb_bas.addAction(self.actionHorizont_lo_koordin_tu_re_is)
        self.menuDarb_bas.addSeparator()
        self.menuDarb_bas.addAction(self.actionProjic_t)
        self.menuDarb_bas.addAction(self.actionProjic_t_kop)
        self.menuDarb_bas.addAction(self.actionProjic_t_no_augstumu_kartes)
        self.menuDarb_bas.addAction(self.actionProjic_t_kop_no_augstumu_kartes)
        self.menuDarb_bas.addSeparator()
        self.menuDarb_bas.addAction(self.actionKontrolpunktu_augstumus)
        self.menuDarb_bas.addAction(self.actionAugstumu_karti)
        self.menuZ_m_t.addAction(self.actionCiparot_zvaigznes)
        self.menuZ_m_t.addSeparator()
        self.menuZ_m_t.addAction(self.actionKalibr_t_kameru)
        self.menuZ_m_t.addSeparator()
        self.menuZ_m_t.addAction(self.actionKontrolpunkti)
        self.menuZ_m_t.addSeparator()
        self.menuZ_m_t.addAction(self.actionIzveidot_augstumu_karti)
        self.menuZ_m_t.addSeparator()
        self.menuZ_m_t.addAction(self.actionProjekcijas_apgabals)
        self.menuZ_m_t.addAction(self.actionKartes_apgabals)
        self.menuZ_m_t.addAction(self.actionKameras_kalibr_cijas_parametri)
        self.menuZ_m_t.addSeparator()
        self.menuZ_m_t.addAction(self.actionUzst_d_t_datumu)
        self.menuZ_m_t.addAction(self.actionUzst_d_t_platumu_garumu_augstumu)
        self.menubar.addAction(self.menuFails.menuAction())
        self.menubar.addAction(self.menuZ_m_t.menuAction())
        self.menubar.addAction(self.menuDarb_bas.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Sudrabaino mākoņu apstrādes programma"))
        self.menuFails.setTitle(_translate("MainWindow", "Fails"))
        self.menuDarb_bas.setTitle(_translate("MainWindow", "Zīmēt"))
        self.menuZ_m_t.setTitle(_translate("MainWindow", "Darbības"))
        self.actionIelas_t_att_lu.setText(_translate("MainWindow", "Ielasīt attēlu"))
        self.actionIelas_t_projektu.setText(_translate("MainWindow", "Ielasīt projektu"))
        self.actionKalibr_t_kameru.setText(_translate("MainWindow", "Kalibrēt kameru"))
        self.actionSaglab_t_projektu.setText(_translate("MainWindow", "Saglabāt projektu"))
        self.actionHorizont_lo_koordin_tu_re_is.setText(_translate("MainWindow", "Horizontālo koordinātu režģis"))
        self.actionCiparot_zvaigznes.setText(_translate("MainWindow", "Ciparot zvaigznes"))
        self.actionAtt_lu.setText(_translate("MainWindow", "Attēls"))
        self.actionProjic_t.setText(_translate("MainWindow", "Projicēt"))
        self.actionProjekcijas_apgabals.setText(_translate("MainWindow", "Projekcijas apgabals"))
        self.actionKartes_apgabals.setText(_translate("MainWindow", "Kartes apgabals"))
        self.actionIelas_t_otro_projektu.setText(_translate("MainWindow", "Ielasīt otro projektu"))
        self.actionProjic_t_kop.setText(_translate("MainWindow", "Projicēt kopā"))
        self.actionKontrolpunkti.setText(_translate("MainWindow", "Kontrolpunkti"))
        self.actionIelas_t_kontrolpunktus.setText(_translate("MainWindow", "Ielasīt kontrolpunktus"))
        self.actionKontrolpunktu_augstumus.setText(_translate("MainWindow", "Kontrolpunktu augstumus"))
        self.actionIzveidot_augstumu_karti.setText(_translate("MainWindow", "Izveidot augstumu karti"))
        self.actionSaglab_t_augstumu_karti.setText(_translate("MainWindow", "Saglabāt augstumu karti"))
        self.actionIelas_t_augstumu_karti.setText(_translate("MainWindow", "Ielasīt augstumu karti"))
        self.actionAugstumu_karti.setText(_translate("MainWindow", "Augstumu karti"))
        self.actionProjic_t_no_augstumu_kartes.setText(_translate("MainWindow", "Projicēt no augstumu kartes"))
        self.actionProjic_t_kop_no_augstumu_kartes.setText(_translate("MainWindow", "Projicēt kopā no augstumu kartes"))
        self.actionKameras_kalibr_cijas_parametri.setText(_translate("MainWindow", "Kameras kalibrācijas parametri"))
        self.actionSaglab_t_projic_to_att_lu_JPG.setText(_translate("MainWindow", "Saglabāt projicēto attēlu JPG"))
        self.actionSaglab_t_projic_to_att_lu_TIFF.setText(_translate("MainWindow", "Saglabāt projicēto attēlu TIFF"))
        self.actionIelas_t_kameru.setText(_translate("MainWindow", "Ielasīt kameru"))
        self.actionSaglab_t_kameru.setText(_translate("MainWindow", "Saglabāt kameru"))
        self.actionUzst_d_t_datumu.setText(_translate("MainWindow", "Uzstādīt datumu"))
        self.actionUzst_d_t_platumu_garumu_augstumu.setText(_translate("MainWindow", "Uzstādīt platumu, garumu, augstumu"))
from mplwidget import MplWidget
