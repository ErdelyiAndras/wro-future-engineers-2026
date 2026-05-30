from __future__ import annotations

import struct
import threading

import serial

from components.Component import Component
from utils import Event, mm

class Arduino(Component):
    _START_BYTE     = 0xAA
    _MSG_SET_TARGET = 0x01
    _MSG_STATE      = 0x02
    _MSG_STOP       = 0x03

    def __init__(self, port: str = '/dev/arduino', baud_rate: int = 115200) -> None:
        super().__init__()
        self._port:       str                  = port
        self._baud_rate:  int                  = baud_rate
        self._serial:     serial.Serial | None = None
        self._write_lock: threading.Lock       = threading.Lock()

        self.on_state: Event = Event()

    def set_target(self, forward_mm: mm, lateral_mm: mm, speed: float) -> None:
        payload = struct.pack('<3f', forward_mm, lateral_mm, speed)
        self._send(self._MSG_SET_TARGET, payload)

    def stop(self) -> None:
        self._send(self._MSG_STOP, b'')

    def _setup(self) -> None:
        self._serial = serial.Serial(self._port, self._baud_rate, timeout = 1)

    def _teardown(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def _run(self) -> None:
        buf = bytearray()
        while self._is_running:
            if not self._serial or not self._serial.is_open:
                break
            try:
                chunk = self._serial.read(64)
            except serial.SerialException:
                chunk = b''
                break
            if not chunk:
                continue
            buf.extend(chunk)
            buf = self._parse(buf)

    def _parse(self, buf: bytearray) -> bytearray:
        while len(buf) >= 4:
            if buf[0] != self._START_BYTE:
                del buf[0]
                continue

            msg_id = buf[1]
            length = buf[2]
            total  = 4 + length

            if len(buf) < total:
                break

            payload  = bytes(buf[3 : 3 + length])
            crc_recv = buf[3 + length]
            crc_calc = self._crc8(bytes([msg_id, length]) + payload)

            if crc_recv != crc_calc:
                del buf[0]
                continue

            self._dispatch(msg_id, payload)
            del buf[:total]

        return buf

    def _dispatch(self, msg_id: int, payload: bytes) -> None:
        if msg_id == self._MSG_STATE and len(payload) == 16:
            heading, speed, steering, distance = struct.unpack('<4f', payload)
            self.on_state(heading, speed, steering, distance)

    def _send(self, msg_id: int, payload: bytes) -> None:
        length = len(payload)
        crc    = self._crc8(bytes([msg_id, length]) + payload)
        frame  = bytes([self._START_BYTE, msg_id, length]) + payload + bytes([crc])
        with self._write_lock:
            if self._serial and self._serial.is_open:
                self._serial.write(frame)

    @staticmethod
    def _crc8(data: bytes) -> int:
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = ((crc << 1) ^ 0x07) if crc & 0x80 else (crc << 1)
                crc &= 0xFF
        return crc
