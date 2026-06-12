#include "Protocol.h"
#include <string.h>

Protocol protocol;

uint8_t Protocol::crc8(const uint8_t* data, uint8_t len, uint8_t init) {
    uint8_t crc = init;
    for (uint8_t i = 0U; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0U; b < 8U; b++) {
            crc = (crc & 0x80) ? ((crc << 1) ^ 0x07) : (crc << 1);
        }
    }
    return crc;
}

void Protocol::dispatch(uint8_t msg_id, const uint8_t* payload, uint8_t length) {
    if (msg_id == Protocol::MSG_SET_TARGET && length == 12) {
        float fwd, lat, spd;
        memcpy(&fwd, payload + 0, 4);
        memcpy(&lat, payload + 4, 4);
        memcpy(&spd, payload + 8, 4);
        if (onSetTarget) {
            onSetTarget(fwd, lat, spd);
        }
    } else if (msg_id == Protocol::MSG_STOP && length == 0) {
        if (onStop) {
            onStop();
        }
    }
}

void Protocol::update() {
    while (Serial.available() && buf_len < BUF_SIZE) {
        buf[buf_len++] = static_cast<uint8_t>(Serial.read());
    }

    uint8_t consumed = 0U;
    while (consumed < buf_len) {
        if (buf[consumed] != Protocol::START_BYTE) {
            consumed++;
            continue;
        }

        if (buf_len - consumed < 4) {
            break;
        }

        uint8_t msg_id = buf[consumed + 1];
        uint8_t length = buf[consumed + 2];
        uint8_t total  = 4U + length;

        if (total > BUF_SIZE) {
            consumed++;
            continue;
        }

        if (buf_len - consumed < total) {
            break;
        }

        const uint8_t* payload  = buf + consumed + 3;
        uint8_t        crc_recv = buf[consumed + 3 + length];

        uint8_t hdr[2]   = {msg_id, length};
        uint8_t crc_calc = crc8(hdr, 2U, 0U);
        crc_calc         = crc8(payload, length, crc_calc);

        if (crc_recv != crc_calc) {
            consumed++;
            continue;
        }

        dispatch(msg_id, payload, length);
        consumed += total;
    }

    if (consumed > 0) {
        buf_len -= consumed;
        if (buf_len > 0) {
            memmove(buf, buf + consumed, buf_len);
        }
    }
}

void Protocol::sendState(float heading_rad, float speed, float steering_rad, float dist_delta_mm) {
    uint8_t payload[16];
    memcpy(payload + 0,  &heading_rad,   4);
    memcpy(payload + 4,  &speed,         4);
    memcpy(payload + 8,  &steering_rad,  4);
    memcpy(payload + 12, &dist_delta_mm, 4);

    uint8_t hdr[2] = {Protocol::MSG_STATE, 16U};
    uint8_t crc    = crc8(hdr, 2U, 0U);
    crc            = crc8(payload, 16, crc);

    uint8_t frame[20];
    frame[0] = Protocol::START_BYTE;
    frame[1] = MSG_STATE;
    frame[2] = 16;
    memcpy(frame + 3, payload, 16);
    frame[19] = crc;

    Serial.write(frame, 20);
}
