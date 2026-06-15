#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>
#include <math.h>

namespace config {

namespace pins{
    static constexpr int PIN_MOTOR_FWD = 5;
    static constexpr int PIN_MOTOR_REV = 6;
    static constexpr int PIN_SERVO     = 9;
    static constexpr int PIN_ENC_A     = 2;
    static constexpr int PIN_ENC_B     = 3;
} // namespace pins

namespace serial {
    static constexpr unsigned long SERIAL_BAUD = 115200UL;
} // namespace serial

namespace movement {
    static constexpr int   MOTOR_DIR         = -1;
    static constexpr float WHEEL_DIAMETER_MM = 43.2f;
    static constexpr float TICKS_PER_MM      = 15.17f;
    static constexpr float MAX_SPEED         = 255.0f;
} // namespace movement

namespace steering {
    static constexpr int   SERVO_CENTER_US = 1050;
    static constexpr int   SERVO_LEFT_US   = 1400;
    static constexpr int   SERVO_RIGHT_US  = 500;
    static constexpr float MAX_STEER_RAD   = 0.4014f;
} // namespace steering

namespace encoder {
    static constexpr int SPEED_SAMPLE_MS = 50;
} // namespace encoder

namespace navigation {
    static constexpr float WHEELBASE_MM           = 89.6251f;
    static constexpr float HEADING_KP             = -2.5f;
    static constexpr float HEADING_KI             = -0.7f;
    static constexpr float HEADING_KD             = -0.155f;
    static constexpr float HEADING_I_CLAMP        = 0.3f;
    static constexpr float HEADING_D_FILTER       = 0.7f;
    static constexpr float HEADING_HOLD_RADIUS_MM = 100.0f;
    static constexpr float ARRIVAL_THRESHOLD_MM   = 10.0f;
    static constexpr float ACCEL_RADIUS_MM        = 200.0f;
    static constexpr float DECEL_RADIUS_MM        = 500.0f;
    static constexpr float MIN_SPEED              = 80.0f;
    static const     float MIN_TURNING_RADIUS_MM  = (WHEELBASE_MM / tanf(steering::MAX_STEER_RAD));
} // namespace navigation

namespace reporting {
    static constexpr int STATE_REPORT_INTERVAL_MS = 20;
} // namespace reporting

inline float wrapPi(float angle) {
    angle = fmodf(angle + M_PI, 2.0f * M_PI);
    if (angle < 0) {
        angle += 2.0f * M_PI;
    }
    return angle - M_PI;
}

} // namespace config

#endif // CONFIG_H
