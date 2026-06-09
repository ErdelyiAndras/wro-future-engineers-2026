FQBN = arduino:avr:nano:cpu=atmega328
PORT = /dev/arduino
SKETCH = src/arduino

flash: compile upload

compile:
	arduino-cli compile --fqbn $(FQBN) $(SKETCH)

upload:
	arduino-cli upload -p $(PORT) --fqbn $(FQBN) $(SKETCH)
