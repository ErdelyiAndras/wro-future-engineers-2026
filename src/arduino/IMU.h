#ifndef IMU_H
#define IMU_H

#include <Adafruit_BNO055.h>

class IMUController {
public:
    bool  begin();
    void  update();
    float getHeadingRad() const;

private:
    Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
    float heading_rad        = 0.0f;
    float heading_offset_rad = 0.0f;
    bool  ready              = false;
};

extern IMUController imuCtrl;

#endif // IMU_H
