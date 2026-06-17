#include "Motor.h"
#include "Config.h"

Motor motor;

void Motor::begin() {
    pinMode(config::pins::PIN_MOTOR_FWD, OUTPUT);
    pinMode(config::pins::PIN_MOTOR_REV, OUTPUT);
    setPwm(0.0f);
}

void Motor::setTarget(float speed) {
    if (stopped &&
        fabsf(speed) > 0.0f &&
        fabsf(speed) < config::navigation::MIN_SPEED) {
        current        = copysignf(config::navigation::MIN_SPEED, speed);
        kickstarting   = true;
        last_update_ms = millis();
    }
    target  = speed;
    stopped = false;
}

void Motor::stop() {
    target       = 0.0f;
    current      = 0.0f;
    stopped      = true;
    kickstarting = false;
    setPwm(0.0f);
}

void Motor::brake() {
    target       = 0.0f;
    current      = 0.0f;
    stopped      = true;
    kickstarting = false;
    analogWrite(config::pins::PIN_MOTOR_FWD, 255);
    analogWrite(config::pins::PIN_MOTOR_REV, 255);
}

void Motor::update() {
    if (stopped) {
        return;
    }

    if (kickstarting) {
        unsigned long now = millis();
        float         dt  = (now - last_update_ms) * 1e-3f;
        last_update_ms    = now;

        float step = config::movement::KICKSTART_DECAY * dt;
        if (fabsf(current) - fabsf(target) <= step) {
            current      = target;
            kickstarting = false;
        } else {
            current -= copysignf(step, current);
        }
    } else {
        current = target;
    }

    setPwm(
        constrain(
            current / config::movement::MAX_SPEED,
            -1.0f,
            1.0f
        ) * config::movement::MOTOR_DIR
    );
}

void Motor::setPwm(float normalized) {
    int pwm = static_cast<int>(fabsf(normalized) * 255.0f);
    pwm = constrain(pwm, 0, 255);

    if (normalized > 0.01f) {
        analogWrite(config::pins::PIN_MOTOR_FWD, pwm);
        analogWrite(config::pins::PIN_MOTOR_REV, 0);
    } else if (normalized < -0.01f) {
        analogWrite(config::pins::PIN_MOTOR_FWD, 0);
        analogWrite(config::pins::PIN_MOTOR_REV, pwm);
    } else {
        analogWrite(config::pins::PIN_MOTOR_FWD, 0);
        analogWrite(config::pins::PIN_MOTOR_REV, 0);
    }
}
