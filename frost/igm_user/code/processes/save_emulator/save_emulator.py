#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Saves the iceflow network at the end of an IGM run.

The field_inversion may fine-tune the pretrained emulator on the glacier
(nbit_init; 0 in central_europe_esmda*, so there it is the pretrained
network itself), so the inverted fields and the surface velocities they
produce belong to that network. Forward
runs have to load the same network (pretrained_path) to start without an
initialisation shock.
"""

from igm.processes.iceflow.emulate.utils.artifacts import save_emulator_artifact


def initialize(cfg, state):
    pass


def update(cfg, state):
    pass


def finalize(cfg, state):
    path = save_emulator_artifact(cfg.processes.save_emulator.path,
                                  state.iceflow_model)
    print(f"Saved iceflow emulator to {path}")
