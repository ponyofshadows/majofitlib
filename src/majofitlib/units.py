from .base import unit_register
unit_register("magnetic field", "T", multiplier=1, unicode_name="T")

unit_register("resistivity", "Ohm*m", multiplier=1, unicode_name="Ω·m")
unit_register("resistivity", "Ohm*cm", multiplier=1e-2, unicode_name="Ω·cm")
unit_register("resistivity", "mOhm*cm", multiplier=1e-5, unicode_name="mΩ·cm")
unit_register("resistivity", "muOhm*cm", multiplier=1e-8,unicode_name="μΩ·m")


unit_register("carrier concentration", "m^-3", multiplier=1,unicode_name="m⁻³")
unit_register("carrier concentration", "cm^-3", multiplier=1e-6,unicode_name="cm⁻³")

unit_register("mobility", "m^2/(V*s)", multiplier=1,unicode_name="m²/(V·s)")
unit_register("mobility", "cm^2/(V*s)", multiplier=1e-4,unicode_name="cm²/(V·s)")