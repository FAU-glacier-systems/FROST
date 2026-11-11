# Third-party notices

FROST is licensed under the GNU General Public License v3.0 (`LICENSE`).
It includes or adapts the following third-party material, which keeps its own
license.

## Code

### IGM processes (GPL-3.0)

`frost/igm_user/code/processes/clim_1D_3D/` and `smb_1D_3D/` are adapted from
processes of the Instructed Glacier Model (IGM,
https://github.com/instructed-glacier-model/igm), Copyright (C) 2021-2025 IGM
authors, published under the GNU General Public License v3.0, the license of
FROST.

### OGGM (BSD-3-Clause)

`tile_merge_reproject` in `frost/preprocess/create_observation.py` is adapted
from `hugonnet_to_gdir` in `oggm/shop/hugonnet_maps.py` of the Open Global
Glacier Model (OGGM, https://github.com/OGGM/oggm), under this license:

```
Copyright (c) 2014-2021, OGGM e.V. and OGGM Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Fonts (SIL Open Font License 1.1)

`frost/visualization/fonts/`: Inter (Copyright (c) 2016 The Inter Project
Authors, `Inter-OFL.txt`) and JetBrains Mono (Copyright 2020 The JetBrains Mono
Project Authors, `OFL.txt`), unchanged.

## Data

The reference files in `data/raw/` (GLAMOS, WGMS, RGI 7.0) are not covered by
the GPL; their sources, terms and citations are listed in `data/raw/README.md`.
Input data downloaded by the pipeline (OGGM shop, IceBoost, ITS_LIVE, Hugonnet
et al. 2021) are not part of the repository and keep the terms of their
providers.

## Dependencies

FROST is installed together with, but does not include, IGM (GPL-3.0),
TensorFlow and Keras (Apache-2.0), OGGM, NumPy, SciPy, pandas, rasterio
(BSD-3-Clause), xarray and rioxarray (Apache-2.0), matplotlib (PSF-based),
gstools (LGPL-3.0) and further packages under MIT or BSD licenses (see
`setup.py`, `environment.yml`). All are compatible with GPL-3.0.
