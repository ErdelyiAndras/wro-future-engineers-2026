#include "Steering.h"
#include "Config.h"

SteeringController steering;

void SteeringController::begin() {
    servo.attach(config::pins::PIN_SERVO);
    center();
}

void SteeringController::setAngle(float steering_rad) {
    current_rad     = constrain(
                        steering_rad,
                        -config::steering::MAX_STEER_RAD,
                        config::steering::MAX_STEER_RAD
                    );
    float t         = current_rad / config::steering::MAX_STEER_RAD;
    int   us_offset = static_cast<int>(t * (
        t >= 0
        ? (config::steering::SERVO_LEFT_US   - config::steering::SERVO_CENTER_US)
        : (config::steering::SERVO_CENTER_US - config::steering::SERVO_RIGHT_US)
    ));
    servo.writeMicroseconds(config::steering::SERVO_CENTER_US + us_offset);
}

void SteeringController::center() {
    setAngle(0.0f);
}

float SteeringController::getAngleRad() const {
    return current_rad;
}
