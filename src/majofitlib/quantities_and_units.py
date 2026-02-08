"""
Only consider units in SI
"""
from .base import quantity_unit_register


quantity_unit_register("charge", "statC", multiplier=1, unicode_name="statC")
quantity_unit_register("charge", "C", multiplier=1 / 11.602176634E-19, unicode_name="C")


quantity_unit_register("magnetic field", "G", multiplier=1, unicode_name="G")
quantity_unit_register("magnetic field", "T", multiplier=1E-4, unicode_name="T")

quantity_unit_register("resistivity", "Ohm*m", multiplier=1, unicode_name="Ω·m")
quantity_unit_register("resistivity", "Ohm*cm", multiplier=1E-2, unicode_name="Ω·cm")
quantity_unit_register("resistivity", "mOhm*cm", multiplier=1E-5, unicode_name="mΩ·cm")
quantity_unit_register("resistivity", "muOhm*cm", multiplier=1E-8,unicode_name="μΩ·m")


quantity_unit_register("carrier concentration", "m^-3", multiplier=1,unicode_name="m⁻³")
quantity_unit_register("carrier concentration", "cm^-3", multiplier=1E-6,unicode_name="cm⁻³")

quantity_unit_register("mobility", "m^2/(V*s)", multiplier=1,unicode_name="m²/(V·s)")
quantity_unit_register("mobility", "cm^2/(statV*s)", multiplier=1E-4,unicode_name="cm²/(V·s)")