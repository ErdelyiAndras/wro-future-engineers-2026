#ifndef STEERING_H
#define STEERING_H

#include <Servo.h>

class SteeringController {
public:
    void  begin();
    void  setAngle(float steering_rad);
    void  center();
    float getAngleRad() const;

private:
    Servo servo;
    float current_rad = 0.0f;
};

extern SteeringController steering;

#endif // STEERING_H
