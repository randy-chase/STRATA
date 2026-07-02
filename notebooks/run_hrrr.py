# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util
    import os
    import sys
    from pathlib import Path

    import marimo as mo

    root_path = mo.notebook_dir().parent.as_posix()
    sys.path.append(root_path)

    missing = []
    if importlib.util.find_spec("s3fs") is None:
        missing.append("s3fs")
    if importlib.util.find_spec("pyproj") is None:
        missing.append("pyproj")
    if missing:
        raise ModuleNotFoundError(
            "Missing optional dependencies for HRRR access: "
            f"{missing}. Install with `pip install {' '.join(missing)}`."
        )

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    return Path, mo


@app.cell
def _():
    import dotenv

    dotenv.load_dotenv()

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import xarray as xr
    import zarr
    from pyproj import CRS, Transformer

    import s3fs
    from screamcast.earth2studio_wrappers import ScreamcastModel
    from screamcast.thermodynamics import potential_temperature_from_temperature
    from screamcast.vertical_interpolation import log_pressure_interpolate

    return (
        CRS,
        ScreamcastModel,
        Transformer,
        log_pressure_interpolate,
        np,
        plt,
        potential_temperature_from_temperature,
        s3fs,
        torch,
        xr,
        zarr,
    )


@app.cell
def _(Path, np):
    checkpoint_path = "../pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128_4stepft/output/best.pth"
    tile_size = 256
    center_lat = 39.0
    center_lon = -97.0
    n_steps = 12
    analysis_time = np.datetime64("2026-07-01T12:00:00")
    vertical_coord_path = (
        Path("../data/scream_vertical_coordinate.nc").resolve().as_posix()
    )
    return (
        analysis_time,
        center_lat,
        center_lon,
        checkpoint_path,
        n_steps,
        tile_size,
        vertical_coord_path,
    )


@app.cell
def _(CRS, Transformer, np, s3fs, zarr):
    HRRR_PR_LEVELS_HPA = [
        1000,
        975,
        950,
        925,
        900,
        875,
        850,
        825,
        800,
        775,
        750,
        725,
        700,
        675,
        650,
        625,
        600,
        575,
        550,
        525,
        500,
        475,
        450,
        425,
        400,
        375,
        350,
        325,
        300,
        275,
        250,
        225,
        200,
        175,
        150,
        125,
        100,
        75,
        50,
    ]

    fs = s3fs.S3FileSystem(anon=True)

    def _run_root(prefix: str, yyyymmdd: str, hh: int):
        root = f"s3://hrrrzarr/{prefix}/{yyyymmdd}/{yyyymmdd}_{hh:02d}z_anl.zarr"
        return root

    def _open_array(path: str):
        return zarr.open_array(s3fs.S3Map(root=path, s3=fs, check=False), mode="r")

    def _read_hrrr_var(root: str, level: str, var_name: str, y_slice=None, x_slice=None):
        array_path = f"{root}/{level}/{var_name}/{level}/{var_name}"
        arr = _open_array(array_path)
        if y_slice is None or x_slice is None:
            return np.asarray(arr[:], dtype=np.float32)
        return np.asarray(arr[y_slice, x_slice], dtype=np.float32)

    def _latlon_from_xy(root: str):
        # HRRR CONUS uses Lambert Conformal with spherical Earth radius 6371229 m.
        # Ref: Earth2Studio/Herbie HRRR projection metadata.
        crs = CRS.from_cf(
            {
                "grid_mapping_name": "lambert_conformal_conic",
                "standard_parallel": (38.5, 38.5),
                "latitude_of_projection_origin": 38.5,
                "longitude_of_central_meridian": 262.5,
                "earth_radius": 6371229.0,
            }
        )
        x = _open_array(f"{root}/1000mb/TMP/projection_x_coordinate")[:].astype("float64")
        y = _open_array(f"{root}/1000mb/TMP/projection_y_coordinate")[:].astype("float64")
        xx, yy = np.meshgrid(x, y)
        to_wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = to_wgs84.transform(xx, yy)
        return lat.astype("float32"), lon.astype("float32")

    def _tile_slices_for_center(lat, lon, center_lat, center_lon, tile_size):
        dist = (lat - center_lat) ** 2 + (lon - center_lon) ** 2
        iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
        h, w = lat.shape
        half = tile_size // 2
        y0 = int(np.clip(iy - half, 0, h - tile_size))
        x0 = int(np.clip(ix - half, 0, w - tile_size))
        return slice(y0, y0 + tile_size), slice(x0, x0 + tile_size), (iy, ix)

    def fetch_hrrr_tile(
        analysis_time: np.datetime64,
        center_lat: float,
        center_lon: float,
        tile_size: int,
    ):
        yyyymmdd = str(analysis_time.astype("datetime64[D]")).replace("-", "")
        hh = int(str(analysis_time.astype("datetime64[h]"))[-2:])
        prs_root = _run_root("prs", yyyymmdd, hh)
        sfc_root = _run_root("sfc", yyyymmdd, hh)

        lat, lon = _latlon_from_xy(prs_root)
        y_slice, x_slice, nearest_idx = _tile_slices_for_center(
            lat, lon, center_lat, center_lon, tile_size
        )
        lat_tile = lat[y_slice, x_slice]
        lon_tile = lon[y_slice, x_slice]

        def read_pr(var_name: str):
            stacks = []
            for level in HRRR_PR_LEVELS_HPA:
                arr = _read_hrrr_var(
                    prs_root, f"{level}mb", var_name, y_slice, x_slice
                )
                stacks.append(arr)
            return np.stack(stacks, axis=0)

        temp = read_pr("TMP")
        u = read_pr("UGRD")
        v = read_pr("VGRD")
        z_mid = read_pr("HGT")
        omega = read_pr("VVEL")
        qv = read_pr("SPFH")

        t2m = _read_hrrr_var(sfc_root, "2m_above_ground", "TMP", y_slice, x_slice)
        hgt_sfc = _read_hrrr_var(sfc_root, "surface", "HGT", y_slice, x_slice)
        ps = _read_hrrr_var(sfc_root, "surface", "PRES", y_slice, x_slice)
        landfrac = _read_hrrr_var(sfc_root, "surface", "LAND", y_slice, x_slice)

        return {
            "analysis_time": analysis_time,
            "prs_root": prs_root,
            "sfc_root": sfc_root,
            "nearest_index": nearest_idx,
            "lat_tile": lat_tile.astype("float32"),
            "lon_tile": lon_tile.astype("float32"),
            "temp": temp,
            "u": u,
            "v": v,
            "z_mid": z_mid,
            "omega": omega,
            "qv": qv,
            "t2m": t2m.astype("float32"),
            "ps": ps.astype("float32"),
            "hgt_sfc": hgt_sfc.astype("float32"),
            "landfrac": landfrac.astype("float32"),
        }

    return HRRR_PR_LEVELS_HPA, fetch_hrrr_tile


@app.cell
def _(ScreamcastModel, checkpoint_path):
    model = ScreamcastModel.from_checkpoint(checkpoint_path, bf16=True)
    model = model.cuda()
    model.eval()
    in_coords = dict(model.input_coords())
    out_coords0 = dict(model.output_coords(in_coords))
    print(f"Loaded model from {checkpoint_path}")
    print(f"input channels={len(in_coords['variable'])}, output channels={len(out_coords0['variable'])}")
    return in_coords, model


@app.cell
def _(
    analysis_time,
    center_lat,
    center_lon,
    fetch_hrrr_tile,
    tile_size,
):
    hrrr_tile = fetch_hrrr_tile(
        analysis_time=analysis_time,
        center_lat=center_lat,
        center_lon=center_lon,
        tile_size=tile_size,
    )
    print("Fetched HRRR tile")
    print(f"  prs={hrrr_tile['prs_root']}")
    print(f"  sfc={hrrr_tile['sfc_root']}")
    print(f"  nearest_idx={hrrr_tile['nearest_index']}")
    print(f"  tile lat range=({hrrr_tile['lat_tile'].min():.2f}, {hrrr_tile['lat_tile'].max():.2f})")
    print(f"  tile lon range=({hrrr_tile['lon_tile'].min():.2f}, {hrrr_tile['lon_tile'].max():.2f})")
    return (hrrr_tile,)


@app.cell
def _(
    hrrr_tile,
    in_coords,
    log_pressure_interpolate,
    np,
    potential_temperature_from_temperature,
    torch,
    vertical_coord_path,
    xr,
):
    with xr.open_dataset(vertical_coord_path) as ds_vert:
        hyam_all = ds_vert["hyam"].values.astype("float32")
        hybm_all = ds_vert["hybm"].values.astype("float32")
        if "P0" in ds_vert:
            p0_pa = float(ds_vert["P0"].values)
        else:
            p0_pa = 100000.0

    level_start, level_end, plevel = 8, 32, 1
    level_indices = np.arange(level_start, level_end, plevel, dtype=np.int64)
    hyam = hyam_all[level_indices]
    hybm = hybm_all[level_indices]

    target_pressure = hyam[:, None, None] * p0_pa + hybm[:, None, None] * hrrr_tile["ps"][None]
    src_pressure = (
        np.asarray(
            [
                1000,
                975,
                950,
                925,
                900,
                875,
                850,
                825,
                800,
                775,
                750,
                725,
                700,
                675,
                650,
                625,
                600,
                575,
                550,
                525,
                500,
                475,
                450,
                425,
                400,
                375,
                350,
                325,
                300,
                275,
                250,
                225,
                200,
                175,
                150,
                125,
                100,
                75,
                50,
            ],
            dtype=np.float32,
        )[:, None, None]
        * 100.0
    )

    def v_interp(values_3d):
        src_vals = torch.from_numpy(values_3d.astype("float32"))
        src_p = torch.from_numpy(np.broadcast_to(src_pressure, values_3d.shape).astype("float32"))
        tgt_p = torch.from_numpy(target_pressure.astype("float32"))
        return log_pressure_interpolate(
            src_values=src_vals,
            src_pressure=src_p,
            target_pressure=tgt_p,
            axis=0,
        ).numpy()

    temp_on_scream = v_interp(hrrr_tile["temp"])
    u_on_scream = v_interp(hrrr_tile["u"])
    v_on_scream = v_interp(hrrr_tile["v"])
    zmid_on_scream = v_interp(hrrr_tile["z_mid"])
    omega_on_scream = v_interp(hrrr_tile["omega"])
    qv_on_scream = v_interp(hrrr_tile["qv"])
    theta_on_scream = potential_temperature_from_temperature(
        torch.from_numpy(temp_on_scream),
        torch.from_numpy(target_pressure.astype("float32")),
        p0_pa=float(p0_pa),
    ).numpy()

    channels = {}
    for i, lev in enumerate(level_indices):
        channels[f"PotentialTemperature_{lev}"] = theta_on_scream[i]
        channels[f"U_{lev}"] = u_on_scream[i]
        channels[f"V_{lev}"] = v_on_scream[i]
        channels[f"z_mid_{lev}"] = zmid_on_scream[i]
        channels[f"omega_{lev}"] = omega_on_scream[i]
        channels[f"qv_{lev}"] = qv_on_scream[i]
    channels["T_2m"] = hrrr_tile["t2m"]

    input_var_names = list(in_coords["variable"])
    x_np = np.stack([channels[name] for name in input_var_names], axis=0)[None, :, None]

    # HRRR-derived static forcing for this tile.
    phis = hrrr_tile["hgt_sfc"] * np.float32(9.80665)
    sgh30 = np.zeros_like(phis, dtype=np.float32)
    landfrac = hrrr_tile["landfrac"].astype(np.float32)
    static_forcing = np.stack([phis, sgh30, landfrac], axis=0)[:, None]

    stats = {
        "theta": (float(theta_on_scream.min()), float(theta_on_scream.max())),
        "qv": (float(qv_on_scream.min()), float(qv_on_scream.max())),
        "u": (float(u_on_scream.min()), float(u_on_scream.max())),
        "v": (float(v_on_scream.min()), float(v_on_scream.max())),
    }
    print(f"Assembled input tensor shape={x_np.shape}")
    for key, (_vmin, _vmax) in stats.items():
        print(f"  {key} range=({_vmin:.4f}, {_vmax:.4f})")

    return static_forcing, x_np


@app.cell
def _(hrrr_tile, model, np, static_forcing, tile_size, torch):
    _device = next(model.parameters()).device
    lat_tile = hrrr_tile["lat_tile"].astype("float32")
    lon_tile = hrrr_tile["lon_tile"].astype("float32")

    model.set_latlon(
        lat_deg=torch.from_numpy(lat_tile[None]).to(_device),
        lon_deg=torch.from_numpy(lon_tile[None]).to(_device),
    )
    model.set_tile_size(tile_size)
    model.compile()

    static_names = list(model._static_forcing_names)
    static_dict = {
        "phis": static_forcing[0, 0],
        "sgh30": static_forcing[1, 0],
        "landfrac": static_forcing[2, 0],
    }
    static_tensor = np.stack([static_dict[name] for name in static_names], axis=0)[:, None]
    model.register_buffer(
        "static_forcing",
        torch.from_numpy(static_tensor).to(_device, dtype=torch.float32),
        persistent=False,
    )
    return


@app.cell
def _(hrrr_tile, in_coords, model, n_steps, np, torch, x_np):
    _device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x_np).to(_device)
    coords_model = dict(in_coords)
    coords_model["face"] = np.array([0], dtype=np.int64)
    coords_model["x"] = np.arange(x_np.shape[-2], dtype=np.int64)
    coords_model["y"] = np.arange(x_np.shape[-1], dtype=np.int64)
    coords_model["batch"] = np.empty(1)
    coords_model["time"] = np.array([hrrr_tile["analysis_time"]])

    dt_minutes = model.dt / np.timedelta64(1, "s") / 60
    all_outputs = []
    out_coords = None
    with torch.inference_mode():
        iterator = model.create_iterator(x_tensor, coords_model)
        next(iterator)
        for step in range(n_steps):
            out_tensor, out_coords = next(iterator)
            all_outputs.append(out_tensor.detach().cpu().numpy())
            print(f"step {step + 1}/{n_steps}", end="\r")
    print(f"\nForecast done: {n_steps} steps ({dt_minutes * n_steps / 60:.1f} h)")
    return all_outputs, dt_minutes, out_coords


@app.cell
def _(in_coords, mo, plt, x_np):
    _all_vars = list(in_coords["variable"])
    qv_var = [v for v in _all_vars if v.startswith("qv_")][-1]
    _vi = _all_vars.index(qv_var)
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _im = _ax.imshow(x_np[0, _vi, 0], cmap="Blues", origin="lower")
    _ax.set_title(f"HRRR IC mapped to SCREAM channels: {qv_var}", fontsize=10)
    _ax.axis("off")
    _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return (qv_var,)


@app.cell
def _(mo, out_coords, qv_var):
    var_selector = mo.ui.dropdown(
        options=list(out_coords["variable"]),
        value=qv_var,
        label="Variable",
    )
    var_selector
    return (var_selector,)


@app.cell
def _(all_outputs, dt_minutes, mo, out_coords, plt, var_selector):
    import tempfile

    from matplotlib.animation import FuncAnimation

    _out_var_names = list(out_coords["variable"])
    _vi = _out_var_names.index(var_selector.value)
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.axis("off")
    _vmin = min(arr[0, _vi, 0].min() for arr in all_outputs)
    _vmax = max(arr[0, _vi, 0].max() for arr in all_outputs)
    if "omega" in var_selector.value:
        _vlim = max(abs(_vmin), abs(_vmax))
        _vmin, _vmax = -_vlim, _vlim
    _im = _ax.imshow(
        all_outputs[0][0, _vi, 0],
        cmap="RdBu_r",
        origin="lower",
        vmin=_vmin,
        vmax=_vmax,
    )
    _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _title = _ax.set_title("")

    def _update(step):
        _im.set_data(all_outputs[step][0, _vi, 0])
        _title.set_text(f"{var_selector.value}  +{dt_minutes * (step + 1) / 60:.1f} h")
        return [_im, _title]

    _anim = FuncAnimation(_fig, _update, frames=len(all_outputs), blit=True)
    _tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    _anim.save(_tmp.name, writer="ffmpeg", fps=5)
    plt.close(_fig)
    mo.video(open(_tmp.name, "rb").read())
    return


if __name__ == "__main__":
    app.run()
