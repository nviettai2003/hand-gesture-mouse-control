# Makefile

CXX ?= g++
CXXFLAGS := -Wall -O2 -I. \
    $(shell pkg-config --cflags opencv4) \
    $(shell pkg-config --cflags libcamera)

LDFLAGS := \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
    $(shell pkg-config --libs libcamera) \
    -ltensorflow-lite \
    -lpthread

TARGET = FINAL

SRCS = main.cpp \
       camera/camera.cpp \
       models/anchors.cpp \
       models/palm.cpp \
       models/hand_landmark.cpp \
       mouse/mouse_control.cpp \
       tracking/roi_tracker.cpp \
       app/capture_worker.cpp \
       app/inference_worker.cpp \
       app/renderer.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
