#ifndef MOUSE_CONTROL_H
#define MOUSE_CONTROL_H

#include <linux/uinput.h>

class MouseController {
public:
    MouseController();
    ~MouseController();
    bool init();
    void destroy();
    void move_absolute(int x, int y);
    
    void click_right();
    void press_left();   
    void release_left(); 

private:
    int fd;
    void emit(int type, int code, int val);
};
#endif