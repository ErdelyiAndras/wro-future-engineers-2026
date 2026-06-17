#ifndef MOTOR_H
#define MOTOR_H
#include <Arduino.h>

class Motor {
public:
    void begin();
    void setTarget(float speed);
    void stop();
    void brake();
    void update();

private:
    float         target         = 0.0f;
    float         current        = 0.0f;
    bool          stopped        = true;
    bool          kickstarting   = false;
    unsigned long last_update_ms = 0;

    void setPwm(float normalized);
};

extern Motor motor;

#endif // MOTOR_H
