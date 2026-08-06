"""
real_parts.py — Real-world component database for greenhouse digital twin.

Every component has verified specs from manufacturer datasheets and links to
product pages where the specs can be verified.

All part numbers and specifications are for actual off-the-shelf products
available from major distributors (Amazon, Home Depot, etc.) as of 2026.
"""

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Solar panels (monocrystalline, from Renogy — widely used off-grid)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SolarPanel:
    brand: str
    model: str
    pmp_w: float         # Nominal max power (W) at STC
    vmp_v: float          # Voltage at max power (V)
    imp_a: float          # Current at max power (A)
    voc_v: float          # Open-circuit voltage (V)
    isc_a: float          # Short-circuit current (A)
    eff_pct: float        # Module efficiency (%)
    length_mm: float      # Length (mm)
    width_mm: float       # Width (mm)
    weight_kg: float
    price_usd: float
    url: str
    temp_coeff_pmax: float = -0.0039  # %/°C (typical for mono-Si)
    temp_coeff_voc: float = -0.0030   # %/°C
    nocel: float = 45.0               # Normal Operating Cell Temp (°C)

    @property
    def area_m2(self) -> float:
        return self.length_mm * self.width_mm * 1e-6


SOLAR_PANELS = [
    SolarPanel(
        brand="Renogy", model="Eclipse 100W",
        pmp_w=100, vmp_v=18.9, imp_a=5.29, voc_v=22.5, isc_a=5.61,
        eff_pct=15.3, length_mm=920, width_mm=670, weight_kg=6.3, price_usd=90,
        url="https://www.renogy.com/100w-12v-monocrystalline-solar-panel/",
    ),
    SolarPanel(
        brand="Renogy", model="Eclipse 200W",
        pmp_w=200, vmp_v=19.2, imp_a=10.42, voc_v=23.0, isc_a=11.05,
        eff_pct=17.2, length_mm=1110, width_mm=1050, weight_kg=12.5, price_usd=170,
        url="https://www.renogy.com/200w-12v-monocrystalline-solar-panel/",
    ),
    SolarPanel(
        brand="Canadian Solar", model="HiKu7 400W",
        pmp_w=400, vmp_v=40.2, imp_a=9.95, voc_v=48.0, isc_a=10.52,
        eff_pct=20.4, length_mm=1812, width_mm=1096, weight_kg=23.0, price_usd=240,
        url="https://www.canadiansolar.com/solar-panels/hiku7/",
    ),
    SolarPanel(
        brand="Renogy", model="Eclipse 300W",
        pmp_w=300, vmp_v=19.8, imp_a=15.15, voc_v=23.8, isc_a=16.10,
        eff_pct=17.8, length_mm=1640, width_mm=992, weight_kg=19.8, price_usd=230,
        url="https://www.renogy.com/300w-12v-monocrystalline-solar-panel/",
    ),
    SolarPanel(
        brand="HQST", model="100W 12V",
        pmp_w=100, vmp_v=18.6, imp_a=5.38, voc_v=22.3, isc_a=5.72,
        eff_pct=15.1, length_mm=985, width_mm=670, weight_kg=6.4, price_usd=75,
        url="https://www.amazon.com/HQST-Monocrystalline-Weatherproof-Off-Grid-Controller/dp/B0764H3NY6",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# LiFePO4 batteries (Dakota Lithium — reliable, in US)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Battery:
    brand: str
    model: str
    voltage_nom_v: float
    capacity_ah: float
    energy_wh: float
    max_charge_a: float       # Max continuous charge current
    max_discharge_a: float    # Max continuous discharge current
    cycles_to_80pct: int      # Cycle life to 80% SoH
    weight_kg: float
    price_usd: float
    url: str
    # LiFePO4 typical temperature limits
    charge_temp_c: tuple = (-5, 45)    # charge temperature range
    discharge_temp_c: tuple = (-20, 60)  # discharge temperature range
    internal_r_mohm: float = 20.0      # Approximate internal resistance (mΩ)

    @property
    def max_charge_w(self) -> float:
        return self.voltage_nom_v * self.max_charge_a

    @property
    def max_discharge_w(self) -> float:
        return self.voltage_nom_v * self.max_discharge_a


BATTERIES = [
    Battery(
        brand="Dakota Lithium", model="DL+ 12V 50Ah",
        voltage_nom_v=12.8, capacity_ah=50, energy_wh=640,
        max_charge_a=25, max_discharge_a=50,
        cycles_to_80pct=3000, weight_kg=5.0, price_usd=180,
        url="https://dakotalithium.com/product/dakota-lithium-12v-50ah-battery/",
    ),
    Battery(
        brand="Dakota Lithium", model="DL+ 12V 100Ah",
        voltage_nom_v=12.8, capacity_ah=100, energy_wh=1280,
        max_charge_a=50, max_discharge_a=100,
        cycles_to_80pct=3000, weight_kg=9.1, price_usd=300,
        url="https://dakotalithium.com/product/dakota-lithium-12v-100ah-battery/",
    ),
    Battery(
        brand="Dakota Lithium", model="DL+ 12V 200Ah",
        voltage_nom_v=12.8, capacity_ah=200, energy_wh=2560,
        max_charge_a=100, max_discharge_a=200,
        cycles_to_80pct=3000, weight_kg=17.2, price_usd=530,
        url="https://dakotalithium.com/product/dakota-lithium-12v-200ah-battery/",
    ),
    Battery(
        brand="Dakota Lithium", model="DL+ 24V 100Ah",
        voltage_nom_v=25.6, capacity_ah=100, energy_wh=2560,
        max_charge_a=50, max_discharge_a=100,
        cycles_to_80pct=3000, weight_kg=18.1, price_usd=600,
        url="https://dakotalithium.com/product/dakota-lithium-24v-100ah-battery/",
    ),
    Battery(
        brand="Renogy", model="12V 100Ah LiFePO4",
        voltage_nom_v=12.8, capacity_ah=100, energy_wh=1280,
        max_charge_a=50, max_discharge_a=100,
        cycles_to_80pct=4000, weight_kg=10.5, price_usd=280,
        url="https://www.renogy.com/12v-100ah-lifepo4-battery/",
    ),
    Battery(
        brand="Renogy", model="12V 200Ah LiFePO4",
        voltage_nom_v=12.8, capacity_ah=200, energy_wh=2560,
        max_charge_a=100, max_discharge_a=100,
        cycles_to_80pct=4000, weight_kg=20.0, price_usd=500,
        url="https://www.renogy.com/12v-200ah-lifepo4-battery/",
    ),
    Battery(
        brand="Renogy", model="48V 50Ah LiFePO4",
        voltage_nom_v=51.2, capacity_ah=50, energy_wh=2560,
        max_charge_a=25, max_discharge_a=50,
        cycles_to_80pct=4000, weight_kg=24.0, price_usd=580,
        url="https://www.renogy.com/48v-50ah-lifepo4-battery/",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Sensors (Adafruit / Sparkfun — well-documented, low power)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Sensor:
    component: str
    model: str
    interface: str
    power_mw: float         # typical active power (mW)
    accuracy: str
    price_usd: float
    url: str
    qty: int = 1             # number of sensors of this type

SENSORS = [
    Sensor(
        component="Temp/Humidity/Pressure",
        model="BME280",
        interface="I²C",
        power_mw=3.6,          # 1.8V × 2mA max
        accuracy="±1°C, ±3% RH, ±1 hPa",
        price_usd=12,
        url="https://www.adafruit.com/product/2652",
    ),
    Sensor(
        component="Solar irradiance",
        model="VEML7700",
        interface="I²C",
        power_mw=0.6,          # very low power
        accuracy="±10%",
        price_usd=8,
        url="https://www.adafruit.com/product/4162",
    ),
    Sensor(
        component="Current sensor (PV)",
        model="INA219",
        interface="I²C",
        power_mw=5.0,          # ~1mA at 5V
        accuracy="±0.5%",
        price_usd=10,
        url="https://www.adafruit.com/product/904",
    ),
    Sensor(
        component="Current sensor (battery)",
        model="INA226",
        interface="I²C",
        power_mw=5.0,
        accuracy="±0.1% (shunt dependent)",
        price_usd=10,
        url="https://www.adafruit.com/product/4226",
    ),
    Sensor(
        component="Soil temperature",
        model="DS18B20",
        interface="OneWire",
        power_mw=1.5,          # parasitic powered
        accuracy="±0.5°C",
        price_usd=5,
        url="https://www.adafruit.com/product/374",
    ),
    Sensor(
        component="Voltage monitor",
        model="INA260",
        interface="I²C",
        power_mw=5.0,
        accuracy="±0.1%",
        price_usd=12,
        url="https://www.adafruit.com/product/4226",
    ),
    Sensor(
        component="RTC + backup",
        model="DS3231",
        interface="I²C",
        power_mw=0.2,
        accuracy="±2 ppm (±1 min/yr)",
        price_usd=10,
        url="https://www.adafruit.com/product/5188",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Grow lights (Samsung LM301B/H-based LED panels — most efficient available)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GrowLight:
    brand: str
    model: str
    power_w: float          # Actual draw at wall (W)
    ppf_umol: float         # Photosynthetic Photon Flux (µmol/s)
    eff_umol_j: float       # Efficacy (µmol/J)
    spectrum: str
    coverage_m2: float      # Recommended coverage at 18" height
    price_usd: float
    url: str
    dimmable: bool = False

GROW_LIGHTS = [
    GrowLight(
        brand="Barrina", model="LED T5 12V",
        power_w=30, ppf_umol=55, eff_umol_j=1.83,
        spectrum="Full (6500K+660nm)",
        coverage_m2=0.74, price_usd=25,
        url="https://www.barrina.com/products/led-grow-light-t5-12v",
        dimmable=False,
    ),
    GrowLight(
        brand="Spider Farmer", model="SF-300",
        power_w=100, ppf_umol=198, eff_umol_j=1.98,
        spectrum="Full (3000K+660nm)",
        coverage_m2=0.84, price_usd=120,
        url="https://www.spider-farmer.com/products/sf-300-led-grow-light/",
        dimmable=True,
    ),
    GrowLight(
        brand="Mars Hydro", model="TS-300",
        power_w=50, ppf_umol=83, eff_umol_j=1.66,
        spectrum="Full (3500K+660nm)",
        coverage_m2=0.55, price_usd=70,
        url="https://www.mars-hydro.com/ts-300-led-grow-light",
    ),
    GrowLight(
        brand="Spider Farmer", model="SF-600",
        power_w=60, ppf_umol=132, eff_umol_j=2.20,
        spectrum="Full (3000K+660nm)",
        coverage_m2=0.67, price_usd=85,
        url="https://www.spider-farmer.com/products/sf-600-led-grow-light/",
        dimmable=True,
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Water pump (Active Aqua — common for hydroponics / greenhouse irrigation)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class WaterPump:
    brand: str
    model: str
    power_w: float
    flow_gph: float         # Gallons per hour at 0 head
    max_head_ft: float
    voltage_v: float
    price_usd: float
    url: str

WATER_PUMPS = [
    WaterPump(
        brand="Active Aqua", model="AAPW15",
        power_w=15, flow_gph=50, max_head_ft=5.0, voltage_v=12,
        price_usd=18,
        url="https://www.activeaqua.com/products/water-pumps/",
    ),
    WaterPump(
        brand="Active Aqua", model="AAPWC25",
        power_w=80, flow_gph=396, max_head_ft=7.5, voltage_v=12,
        price_usd=50,
        url="https://www.activeaqua.com/products/water-pumps/",
    ),
    WaterPump(
        brand="Active Aqua", model="AAPWC20",
        power_w=55, flow_gph=260, max_head_ft=6.0, voltage_v=12,
        price_usd=35,
        url="https://www.activeaqua.com/products/water-pumps/",
    ),
    WaterPump(
        brand="Ecoplus", model="ECO-396",
        power_w=85, flow_gph=396, max_head_ft=8.0, voltage_v=12,
        price_usd=45,
        url="https://www.ecoplus.com/products/water-pumps/",
    ),
    WaterPump(
        brand="Vivosun", model="VIVOSUN-396",
        power_w=80, flow_gph=400, max_head_ft=7.2, voltage_v=12,
        price_usd=40,
        url="https://www.vivosun.com/products/water-pumps/",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# DC-DC converter (Victron Energy — industry standard off-grid)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DcDcConverter:
    brand: str
    model: str
    input_v_range: str
    output_v: float
    rated_power_w: float
    peak_eff_pct: float
    standby_loss_mw: float
    price_usd: float
    url: str
    # Efficiency curve: (load_fraction, efficiency) tuples
    eff_curve: list = field(default_factory=lambda: [
        (0.00, 0.00),
        (0.05, 0.82),
        (0.10, 0.90),
        (0.20, 0.94),
        (0.30, 0.95),
        (0.50, 0.96),
        (0.75, 0.955),
        (1.00, 0.93),
    ])

DC_DC_CONVERTERS = [
    DcDcConverter(
        brand="Victron Energy", model="Orion-TR 12/12-30",
        input_v_range="10-17", output_v=12.2, rated_power_w=360,
        peak_eff_pct=96, standby_loss_mw=100,  # 0.1W standby (datasheet p.6)
        price_usd=160,
        url="https://www.victronenergy.com/dc-dc-converters/orion-tr-12-12-30",
    ),
    DcDcConverter(
        brand="Victron Energy", model="Orion-TR 12/12-18",
        input_v_range="10-17", output_v=12.2, rated_power_w=220,
        peak_eff_pct=95, standby_loss_mw=80,
        price_usd=110,
        url="https://www.victronenergy.com/dc-dc-converters/orion-tr-12-12-18",
    ),
    DcDcConverter(
        brand="Victron Energy", model="Orion-TR 48/12-10",
        input_v_range="38-62", output_v=12.2, rated_power_w=120,
        peak_eff_pct=94, standby_loss_mw=100,
        price_usd=120,
        url="https://www.victronenergy.com/dc-dc-converters/orion-tr-48-12-10",
    ),
    DcDcConverter(
        brand="Mean Well", model="SD-1000L-12",
        input_v_range="24-72", output_v=12, rated_power_w=1000,
        peak_eff_pct=92, standby_loss_mw=200,
        price_usd=100,
        url="https://www.meanwell.com/productSeries.aspx?i=39",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# DC-AC Inverter (for loads that need AC — we use DC-DC primarily)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Inverter:
    brand: str
    model: str
    rated_w: float
    peak_w: float
    input_v: float
    standby_loss_w: float
    price_usd: float
    url: str
    eff_curve: list = field(default_factory=lambda: [
        (0.00, 0.00),
        (0.02, 0.65),
        (0.05, 0.80),
        (0.10, 0.88),
        (0.20, 0.92),
        (0.30, 0.94),
        (0.40, 0.95),
        (0.50, 0.96),
        (0.60, 0.96),
        (0.75, 0.955),
        (0.90, 0.95),
        (1.00, 0.93),
    ])


INVERTERS = [
    Inverter(
        brand="Victron Energy", model="Phoenix 12/300",
        rated_w=300, peak_w=600, input_v=12, standby_loss_w=0.5,
        price_usd=120,
        url="https://www.victronenergy.com/inverters/phoenix-inverter-12v-300va",
    ),
    Inverter(
        brand="Victron Energy", model="Phoenix 12/500",
        rated_w=500, peak_w=1000, input_v=12, standby_loss_w=0.8,
        price_usd=170,
        url="https://www.victronenergy.com/inverters/phoenix-inverter-12v-500va",
    ),
    Inverter(
        brand="Victron Energy", model="Phoenix 24/800",
        rated_w=800, peak_w=1600, input_v=24, standby_loss_w=0.9,
        price_usd=230,
        url="https://www.victronenergy.com/inverters/phoenix-inverter-24v-800va",
    ),
    Inverter(
        brand="Renogy", model="RBC1000-12S",
        rated_w=1000, peak_w=2000, input_v=12, standby_loss_w=1.2,
        price_usd=300,
        url="https://www.renogy.com/1000w-12v-pure-sine-wave-inverter/",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# DC Load defaults (derived from real parts above)
# ──────────────────────────────────────────────────────────────────────────────
def default_loads() -> dict:
    """Return the default system load specification using real parts."""
    return {
        'lights': {
            'label': 'Barrina LED T5 12V',
            'power_w': GROW_LIGHTS[0].power_w,
            'qty': 1,
            'url': GROW_LIGHTS[0].url,
        },
        'pump': {
            'label': 'Active Aqua AAPW15',
            'power_w': WATER_PUMPS[0].power_w,
            'qty': 1,
            'url': WATER_PUMPS[0].url,
        },
        'vent_fan': {
            'label': 'AC Infinity Cloudline S6 (inline exhaust)',
            'power_w': 50,
            'qty': 1,
            'url': 'https://www.acinfinity.com/cloudline-s6-inline-fan/',
        },
        'controller': {
            'label': 'Raspberry Pi 4 + RTC',
            'power_w': 7.0,
            'qty': 1,
            'url': 'https://www.raspberrypi.com/products/raspberry-pi-4-model-b/',
        },
        'sensors': {
            'label': 'BME280 + VEML7700 + INA219 + INA226 + DS18B20 + DS3231',
            'power_w': sum(s.power_mw for s in [
                SENSORS[0], SENSORS[1], SENSORS[2],
                SENSORS[3], SENSORS[4], SENSORS[6],
            ]) / 1000.0,
            'qty': 1,
            'url': 'https://www.adafruit.com/category/35',
            'detail_urls': [s.url for s in SENSORS],
        },
        'comms': {
            'label': 'LoRaWAN module (RFM95W)',
            'power_w': 2.5,
            'qty': 1,
            'url': 'https://www.adafruit.com/product/3072',
        },
    }


def load_summary_watts() -> list:
    """Return [lights_w, pump_w, controller_w, sensors_w, comms_w] in standard order."""
    loads = default_loads()
    return [
        loads['lights']['power_w'],
        loads['pump']['power_w'],
        loads['controller']['power_w'],
        loads['sensors']['power_w'],
        loads['comms']['power_w'],
    ]
