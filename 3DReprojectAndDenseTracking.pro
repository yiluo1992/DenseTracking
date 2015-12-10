TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

# This makes the .cu files appear in your project
OTHER_FILES +=  ./cuda_interface.cu ./cuda_interface.cuh

# CUDA settings <-- may change depending on your system
CUDA_SOURCES += ./cuda_interface.cu
CUDA_SDK = "/usr/local/cuda-7.5/"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda-7.5/"            # Path to cuda toolkit install

# DO NOT EDIT BEYOND THIS UNLESS YOU KNOW WHAT YOU ARE DOING....

SYSTEM_NAME = x64         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_30           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR/include /usr/local/cuda-7.5/samples/common/inc
INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/include/eigen3/ /usr/local/include/pcl-1.8/ /usr/local/include/vtk-6.2

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64/

CUDA_OBJECTS_DIR = ./


# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_cudev.so /usr/local/lib/libopencv_cudaarithm.so

LIBS += /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_cudev.so /usr/local/lib/libopencv_cudaarithm.so
LIBS += /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgcodecs.so
LIBS += /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videoio.so /usr/local/lib/libopencv_calib3d.so

LIBS += /usr/local/lib/libpcl_common.so /usr/local/lib/libpcl_visualization.so /usr/local/lib/libpcl_io.so
LIBS += /usr/local/lib/libpcl_features.so /usr/local/lib/libpcl_search.so

LIBS += -lboost_system -lboost_thread

LIBS += /usr/local/lib/libvtkCommonExecutionModel-6.2.so.1 /usr/local/lib/libvtkCommonDataModel-6.2.so.1 /usr/local/lib/libvtkCommonMath-6.2.so
LIBS += /usr/local/lib/libvtkCommonCore-6.2.so /usr/local/lib/libvtkFiltersSources-6.2.so /usr/local/lib/libvtkRenderingCore-6.2.so
LIBS += /usr/local/lib/libvtkRenderingLOD-6.2.so

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

DISTFILES += \
    cuda_interface.cu \
    cuda_interface.cuh
