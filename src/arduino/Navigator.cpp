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
    reversing       = (speed < 0.0f);
    target_speed    = fabsf(speed);
    x_mm            = 0.0f;
    y_mm            = 0.0f;
    if (state != NavState::PURSUING) {
        current_speed = config::navigation::MIN_SPEED;
    }
    target_x_mm     = forward_mm;
    target_y_mm     = lateral_mm;
    heading_ref_rad = imuCtrl.getHeadingRad();
    in_hold         = false;
    pid_integral    = 0.0f;
    pid_prev_error  = 0.0f;
    pid_derivative  = 0.0f;
    pid_last_ms     = millis();
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
        float motion_heading = heading_local + (reversing ? static_cast<float>(M_PI) : 0.0f);
        error = config::wrapPi(atan2f(dy, dx) - motion_heading);
    }
    uint32_t now_ms = millis();
    float    dt     = (now_ms - pid_last_ms) * 1e-3f;
    pid_last_ms     = now_ms;

    if (dt > 0.0f) {
        pid_integral += error * dt;
        if (fabsf(config::navigation::HEADING_KI) > 1e-6f) {
            float i_limit = config::navigation::HEADING_I_CLAMP /
                            fabsf(config::navigation::HEADING_KI);
            pid_integral  = constrain(pid_integral, -i_limit, i_limit);
        }
    }

    if (dt > 0.0f) {
        float raw_derivative = (error - pid_prev_error) / dt;
        pid_derivative = config::navigation::HEADING_D_FILTER          * pid_derivative +
                         (1.0f - config::navigation::HEADING_D_FILTER) * raw_derivative;
    }
    pid_prev_error = error;

    float steer = config::navigation::HEADING_KP * error +
                  config::navigation::HEADING_KI * pid_integral +
                  config::navigation::HEADING_KD * pid_derivative;

    if (reversing) {
        steer *= config::navigation::REVERSE_STEER_SIGN;
    }

    steering.setAngle(
        constrain(
            steer,
            -config::steering::MAX_STEER_RAD,
            config::steering::MAX_STEER_RAD
        )
    );
    float remaining   = sqrtf(dist_sq);
    float speed_ratio = target_speed / config::movement::MAX_SPEED;
    float decel_t     = fminf(
        remaining / fmaxf(
            config::navigation::DECEL_RADIUS_MM * speed_ratio,
            1.0f
        ),
        1.0f
    );
    float decel_speed = config::navigation::MIN_SPEED +
                        decel_t * (target_speed - config::navigation::MIN_SPEED);
    float desired     = fminf(target_speed, decel_speed);

    if (current_speed < desired) {
        float accel_per_mm = (target_speed - config::navigation::MIN_SPEED) /
                             fmaxf(config::navigation::ACCEL_RADIUS_MM * speed_ratio, 1.0f);
        current_speed = fminf(current_speed + fabsf(dist_delta) * accel_per_mm, desired);
    } else {
        current_speed = desired;
    }

    motor.setTarget(reversing ? -current_speed : current_speed);
}

void Navigator::update() {
    if (state == NavState::PURSUING) {
        doPursuing();
    }
}
