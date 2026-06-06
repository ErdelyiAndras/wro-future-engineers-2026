#include "Encoder.h"
#include "Config.h"

Encoder encoder;

static void isrA() { encoder.isrA(); }
static void isrB() { encoder.isrB(); }

void Encoder::begin() {
    pinMode(config::pins::PIN_ENC_A, INPUT_PULLUP);
    pinMode(config::pins::PIN_ENC_B, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(config::pins::PIN_ENC_A), ::isrA, CHANGE);
    attachInterrupt(digitalPinToInterrupt(config::pins::PIN_ENC_B), ::isrB, CHANGE);
    last_speed_time = millis();
}

void Encoder::isrA() {
    uint8_t a = digitalRead(config::pins::PIN_ENC_A);
    uint8_t b = digitalRead(config::pins::PIN_ENC_B);
    ticks += (a == b) ? 1 : -1;
}

void Encoder::isrB() {
    uint8_t a = digitalRead(config::pins::PIN_ENC_A);
    uint8_t b = digitalRead(config::pins::PIN_ENC_B);
    ticks += (a == b) ? -1 : 1;
}

void Encoder::update() {
    unsigned long now = millis();
    unsigned long dt  = now - last_speed_time;
    if (dt < config::encoder::SPEED_SAMPLE_MS) {
        return;
    }

    noInterrupts();
    int32_t t = ticks;
    interrupts();

    int32_t delta = t - last_speed_ticks;
    speed = ticksToMm(delta) / (dt * 0.001f);

    last_speed_ticks = t;
    last_speed_time  = now;
}

int32_t Encoder::getTicks() const {
    noInterrupts();
    int32_t t = ticks;
    interrupts();
    return t;
}

float Encoder::ticksToMm(int32_t ticks) const {
    return static_cast<float>(ticks) / config::movement::TICKS_PER_MM;
}

float Encoder::getSpeed() const {
    return speed;
}
