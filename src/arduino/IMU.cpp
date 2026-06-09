#include "IMU.h"
#include "Config.h"
#include <math.h>

IMUController imuCtrl;

bool IMUController::begin() {
    if (!bno.begin(OPERATION_MODE_IMUPLUS)) {
        return false;
    }
    bno.setExtCrystalUse(true);
    delay(2000);

    sensors_event_t event;
    bno.getEvent(&event);
    heading_offset_rad = config::wrapPi(event.orientation.x * (M_PI / 180.0f));

    ready = true;
    return true;
}

void IMUController::update() {
    if (!ready) {
        return;
    }
    sensors_event_t event;
    bno.getEvent(&event);
    float deg = event.orientation.x;
    heading_rad = config::wrapPi(deg * (M_PI / 180.0f) - heading_offset_rad);
}

float IMUController::getHeadingRad() const {
    return heading_rad;
}
