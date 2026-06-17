#ifndef NAVIGATOR_H
#define NAVIGATOR_H

#include <Arduino.h>

enum class NavState : uint8_t {
    IDLE,
    PURSUING,
};

class Navigator {
public:
    void setTarget(float forward_mm, float lateral_mm, float speed);
    void stop();
    void update();

private:
    NavState state        = NavState::IDLE;
    float    target_speed = 0.0f;
    bool     reversing    = false;
    int32_t  last_ticks   = 0;

    float x_mm             = 0.0f;
    float y_mm             = 0.0f;
    float current_speed    = 0.0f;
    float target_x_mm      = 0.0f;
    float target_y_mm      = 0.0f;
    float heading_ref_rad  = 0.0f;
    float hold_heading_rad = 0.0f;
    bool  in_hold          = false;

    float    pid_integral   = 0.0f;
    float    pid_prev_error = 0.0f;
    float    pid_derivative = 0.0f;
    uint32_t pid_last_ms    = 0;

    void doPursuing();
};

extern Navigator navigator;

#endif // NAVIGATOR_H
