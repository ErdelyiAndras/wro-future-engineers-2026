#include "Motor.h"
#include "Config.h"

Motor motor;

void Motor::begin() {
    pinMode(config::pins::PIN_MOTOR_FWD, OUTPUT);
    pinMode(config::pins::PIN_MOTOR_REV, OUTPUT);
    setPwm(0.0f);
}

void Motor::setTarget(float speed) {
    target  = speed;
    stopped = false;
}

void Motor::stop() {
    target  = 0.0f;
    stopped = true;
    setPwm(0.0f);
}

void Motor::brake() {
    target  = 0.0f;
    stopped = true;
    analogWrite(config::pins::PIN_MOTOR_FWD, 255);
    analogWrite(config::pins::PIN_MOTOR_REV, 255);
}

void Motor::update() {
    if (stopped) {
        return;
    }
    setPwm(
        constrain(
            target / config::movement::MAX_SPEED,
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
