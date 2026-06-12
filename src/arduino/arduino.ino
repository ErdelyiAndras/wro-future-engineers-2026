#include "Config.h"
#include "Protocol.h"
#include "Encoder.h"
#include "IMU.h"
#include "Motor.h"
#include "Steering.h"
#include "Navigator.h"

static unsigned long last_state_report = 0;
static int32_t       report_ticks_ref  = 0;

static void onSetTarget(float forward_mm, float lateral_mm, float speed) {
    navigator.setTarget(forward_mm, lateral_mm, speed);
}

static void onStop() {
    navigator.stop();
}

void setup() {
    Serial.begin(config::serial::SERIAL_BAUD);

    encoder.begin();
    imuCtrl.begin();
    motor.begin();
    steering.begin();

    protocol.onSetTarget = onSetTarget;
    protocol.onStop      = onStop;

    report_ticks_ref  = encoder.getTicks();
    last_state_report = millis();
}

void loop() {
    encoder.update();
    imuCtrl.update();

    protocol.update();
    navigator.update();
    motor.update();

    unsigned long now = millis();
    if (now - last_state_report >= config::reporting::STATE_REPORT_INTERVAL_MS) {
        last_state_report = now;

        int32_t current_ticks = encoder.getTicks();
        float   delta_mm      = encoder.ticksToMm(current_ticks - report_ticks_ref);
        report_ticks_ref      = current_ticks;

        protocol.sendState(
            imuCtrl.getHeadingRad(),
            encoder.getSpeed(),
            steering.getAngleRad(),
            delta_mm
        );
    }
}
