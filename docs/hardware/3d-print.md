# 3D Print

## Printer Selection

The Bambu Lab P1S FDM printer was selected for this project to enable systematic material experimentation and validation across multiple types of filaments. The printer's fully enclosed thermal chamber provides precise temperature regulation, minimizing warping and ensuring dimensional consistency across production runs.

### Key Specifications

- Fully enclosed heated chamber with active thermal management
- Multi-material capability (PLA, PETG, ASA, ABS)
- High positional accuracy suitable for mechanical assemblies

## Material Strategy

A three-stage material selection approach was employed:

### Prototyping Phase

**Material:** PLA

- Used for rapid iteration and design validation
- Lower processing temperature reduces printer stress during concept testing

### Production Phase

Two materials were selected for the final assembly based on functional requirements:

#### ABS

**Application:** Thermal-critical components (e.g., Raspberry Pi 5 mounting plate)

- Superior heat resistance and thermal stability
- Upper service temperature: ~80–100 °C continuous operation
- ABS was selected to provide a substantial safety margin and ensure thermal resilience under worst-case scenarios without risk of material softening or creep

#### PETG-HF (High-Flow PETG)

**Application:** Functional mechanical assemblies (differential gearbox housing, servo mount)

- Enhanced layer adhesion and impact resistance compared to standard PETG
- Adequate thermal resistance for non-critical zones
- Superior dimensional retention and reduced creep under cyclic loading
- **Justification:** PETG is highly durable and tends to bend slightly rather than shatter under stress. This makes it ideal for parts that endure drops, vibrations, or repeated wear. The layer adhesion is also exceptional and parts are less prone to splitting along weak seams when under load.

## Quality Assurance

All printed components underwent:

- Visual inspection for layer uniformity and surface finish
- Dimensional verification against CAD models (tolerance stack-up analysis)
- Mechanical fit testing of assemblies prior to integration
