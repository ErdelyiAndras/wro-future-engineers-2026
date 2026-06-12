#include "Navigator.h"
#include "Config.h"
#include "IMU.h"
#include "Encoder.h"
#include "Motor.h"
#include "Steering.h"
#include <math.h>

Navigator navigator;

void Navigator::setTarget(float forward_mm, float lateral_mm, float speed) {
    const float r = config::navigation::MIN_TURNING_RADIUS_MM;
    bool in_left  =
        (forward_mm * forward_mm + (lateral_mm - r) * (lateral_mm - r)) < (r * r);
    bool in_right =
        (forward_mm * forward_mm + (lateral_mm + r) * (lateral_mm + r)) < (r * r);
    if (in_left || in_right) {
        state = NavState::IDLE;
        return;
    }

    last_ticks      = encoder.getTicks();
    target_speed    = fabsf(speed);
    x_mm            = 0.0f;
    y_mm            = 0.0f;
    target_x_mm     = forward_mm;
    target_y_mm     = lateral_mm;
    heading_ref_rad = imuCtrl.getHeadingRad();
    in_hold         = false;
    state           = NavState::PURSUING;
}

void Navigator::stop() {
    motor.stop();
    steering.center();
    state = NavState::IDLE;
}

void Navigator::doPursuing() {
    int32_t current_ticks = encoder.getTicks();
    float   dist_delta    = encoder.ticksToMm(current_ticks - last_ticks);
    last_ticks = current_ticks;

    float heading_local = config::wrapPi(imuCtrl.getHeadingRad() - heading_ref_rad);
    x_mm += dist_delta * cosf(heading_local);
    y_mm += dist_delta * sinf(heading_local);

    float dx      = target_x_mm - x_mm;
    float dy      = target_y_mm - y_mm;
    float dist_sq = dx * dx + dy * dy;

    static constexpr float ARRIVAL_SQ     =
        config::navigation::ARRIVAL_THRESHOLD_MM * config::navigation::ARRIVAL_THRESHOLD_MM;
    static constexpr float HOLD_RADIUS_SQ =
        config::navigation::HEADING_HOLD_RADIUS_MM * config::navigation::HEADING_HOLD_RADIUS_MM;

    if (dist_sq < ARRIVAL_SQ) {
        motor.brake();
        steering.center();
        state = NavState::IDLE;
        return;
    }

    float error;
    if (dist_sq < HOLD_RADIUS_SQ) {
        if (!in_hold) {
            hold_heading_rad = heading_local;
            in_hold          = true;
        }
        error = config::wrapPi(hold_heading_rad - heading_local);
    } else {
        error = config::wrapPi(atan2f(dy, dx) - heading_local);
    }
    steering.setAngle(constrain(
        config::navigation::HEADING_KP * error,
        -config::steering::MAX_STEER_RAD,
        config::steering::MAX_STEER_RAD
    ));
    motor.setTarget(target_speed);
}

void Navigator::update() {
    if (state == NavState::PURSUING) {
        doPursuing();
    }
}
