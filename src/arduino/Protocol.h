#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <Arduino.h>

class Protocol {
public:
    static constexpr uint8_t START_BYTE     = 0xAA;
    static constexpr uint8_t MSG_SET_TARGET = 0x01;
    static constexpr uint8_t MSG_STATE      = 0x02;
    static constexpr uint8_t MSG_STOP       = 0x03;

    void (*onSetTarget)(float forward_mm, float lateral_mm, float speed) = nullptr;
    void (*onStop)() = nullptr;

    void update();
    void sendState(float heading_rad, float speed, float steering_rad, float dist_delta_mm);

private:
    static constexpr uint8_t BUF_SIZE = 32U;
    uint8_t buf[BUF_SIZE];
    uint8_t buf_len = 0U;

    static uint8_t crc8(const uint8_t* data, uint8_t len, uint8_t init = 0U);
    void dispatch(uint8_t msg_id, const uint8_t* payload, uint8_t length);
};

extern Protocol protocol;

#endif // PROTOCOL_H
