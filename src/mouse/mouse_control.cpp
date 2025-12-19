#include "mouse_control.h"
#include "../core/app_config.h"
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <sys/ioctl.h>

MouseController::MouseController() : fd(-1) {}
MouseController::~MouseController() { if (fd >= 0) destroy(); }

bool MouseController::init() {
    fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "ERR: Cannot open /dev/uinput. Try sudo.\n";
        return false;
    }
    ioctl(fd, UI_SET_EVBIT, EV_KEY);
    ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);
    ioctl(fd, UI_SET_KEYBIT, BTN_RIGHT);
    
    ioctl(fd, UI_SET_EVBIT, EV_ABS);
    ioctl(fd, UI_SET_ABSBIT, ABS_X);
    ioctl(fd, UI_SET_ABSBIT, ABS_Y);

    struct uinput_abs_setup abs_setup;
    memset(&abs_setup, 0, sizeof(abs_setup));
    abs_setup.code = ABS_X;
    abs_setup.absinfo.minimum = 0;
    abs_setup.absinfo.maximum = SCREEN_WIDTH;
    ioctl(fd, UI_ABS_SETUP, &abs_setup);

    abs_setup.code = ABS_Y;
    abs_setup.absinfo.minimum = 0;
    abs_setup.absinfo.maximum = SCREEN_HEIGHT;
    ioctl(fd, UI_ABS_SETUP, &abs_setup);

    struct uinput_setup usetup;
    memset(&usetup, 0, sizeof(usetup));
    usetup.id.bustype = BUS_USB;
    usetup.id.vendor  = 0x1234; 
    usetup.id.product = 0x5678;
    strcpy(usetup.name, "Hand Tracking Virtual Mouse");

    ioctl(fd, UI_DEV_SETUP, &usetup);
    if (ioctl(fd, UI_DEV_CREATE) < 0) {
        std::cerr << "ERR: Cannot create uinput device.\n";
        close(fd); fd = -1;
        return false;
    }
    return true;
}

void MouseController::emit(int type, int code, int val) {
    if (fd < 0) return;
    struct input_event ie;
    memset(&ie, 0, sizeof(ie));
    ie.type = type; ie.code = code; ie.value = val;
    write(fd, &ie, sizeof(ie));
}

void MouseController::move_absolute(int x, int y) {
    if (fd < 0) return;
    if (x < 0) x = 0; if (y < 0) y = 0;
    if (x > SCREEN_WIDTH) x = SCREEN_WIDTH;
    if (y > SCREEN_HEIGHT) y = SCREEN_HEIGHT;
    emit(EV_ABS, ABS_X, x);
    emit(EV_ABS, ABS_Y, y);
    emit(EV_SYN, SYN_REPORT, 0);
}

void MouseController::click_right() {
    if (fd < 0) return;
    emit(EV_KEY, BTN_RIGHT, 1); emit(EV_SYN, SYN_REPORT, 0);
    emit(EV_KEY, BTN_RIGHT, 0); emit(EV_SYN, SYN_REPORT, 0);
}

void MouseController::press_left() {
    if (fd < 0) return;
    emit(EV_KEY, BTN_LEFT, 1);   
    emit(EV_SYN, SYN_REPORT, 0);
}

void MouseController::release_left() {
    if (fd < 0) return;
    emit(EV_KEY, BTN_LEFT, 0);   
    emit(EV_SYN, SYN_REPORT, 0);
}

void MouseController::destroy() {
    if (fd >= 0) {
        ioctl(fd, UI_DEV_DESTROY);
        close(fd); fd = -1;
    }
}
