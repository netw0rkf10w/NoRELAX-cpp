#-------------------------------------------------
#
# Project created by QtCreator 2014-09-04T22:51:38
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = NoRELAX
TEMPLATE = app

CONFIG += c++11

QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp

# remove possible other optimization flags
QMAKE_CXXFLAGS_RELEASE -= -O
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2

# add the desired -O3 if not present
QMAKE_CXXFLAGS_RELEASE += -O3


#Add HDF5 library
#LIBS += -L/usr/lib/x86_64-linux-gnu/ -lhdf5 -lhdf5_cpp
LIBS += -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5 -lhdf5_cpp
INCLUDEPATH += /usr/include/hdf5/serial

#INCLUDEPATH += /home/khue/Libs/Eigen3.3.3
INCLUDEPATH += Eigen

# For fast projection on simplex
# INCLUDEPATH += /usr/include/gsl
# LIBS += -L/usr/lib -lgsl


## CPLEX
#LIBS += -L/opt/ibm/ILOG/CPLEX_Studio127/cplex/lib/x86-64_linux/static_pic/ -lcplex -lilocplex
#LIBS += -L/opt/ibm/ILOG/CPLEX_Studio127/concert/lib/x86-64_linux/static_pic/ -lconcert
#INCLUDEPATH += /opt/ibm/ILOG/CPLEX_Studio127/cplex/include \
#                /opt/ibm/ILOG/CPLEX_Studio127/concert/include
#DEFINES += IL_STD


##Add OpenCV libraries
#INCLUDEPATH +=  /usr/local/include/opencv\
#                /usr/local/include\
#                /opt/local/include\


#LIBS += -L/usr/local/lib \
#-lopencv_calib3d \
#-lopencv_core \
#-lopencv_features2d \
#-lopencv_flann \
#-lopencv_highgui \
#-lopencv_imgcodecs \
#-lopencv_imgproc \
#-lopencv_ml \
#-lopencv_objdetect \
#-lopencv_optflow \
#-lopencv_photo \
#-lopencv_shape \
#-lopencv_stitching \
#-lopencv_superres \
#-lopencv_ts \
#-lopencv_video \
#-lopencv_videoio \
#-lopencv_videostab

HEADERS += \
    norelax/OpenGM.hpp \
    norelax/outils_io.hpp \
    norelax/PairwiseMRF.hpp \
    norelax/unittest.hpp \
    norelax/cxxopts.hpp \
    norelax/HMRF.hpp \
    norelax/lemmas.hpp \
    norelax/makeunique.hpp \
    norelax/projection.h \
    norelax/DenseMRF.hpp

SOURCES += \
    norelax/outils_io.cpp \
    norelax/PairwiseMRF.cpp \
    norelax/HMRF.cpp \
    norelax/lemmas.cpp \
    norelax/main.cpp \
    norelax/DenseMRF.cpp
