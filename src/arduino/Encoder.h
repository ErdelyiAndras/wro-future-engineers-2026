#ifndef ENCODER_H
#define ENCODER_H
#include <Arduino.h>

class Encoder {
public:
    void begin();
    void update();

    int32_t getTicks() const;
    float   ticksToMm(int32_t ticks) const;
    float   getSpeed() const;

    void isrA();
    void isrB();

private:
    volatile int32_t ticks = 0;

    float         speed            = 0.0f;
    int32_t       last_speed_ticks = 0;
    unsigned long last_speed_time  = 0U;
};

extern Encoder encoder;

#endif // ENCODER_H
