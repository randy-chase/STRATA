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
"""Run a zero-shot HRRR tile forecast with the released STRATA checkpoint.

This script maps one HRRR analysis tile onto SCREAM channel conventions,
interpolates pressure-level fields to SCREAM hybrid levels 8-31, and runs a
short autoregressive rollout.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import s3fs
import torch
import xarray as xr
import zarr
from pyproj import CRS, Transformer

# Ensure repo-local imports resolve even when launched outside repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import screamcast.earth2studio_wrappers as e2w
from screamcast.thermodynamics import potential_temperature_from_temperature
from screamcast.vertical_interpolation import log_pressure_interpolate

HRRR_PR_LEVELS_HPA = (
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
)
GRAVITY = np.float32(9.80665)
STATIC_FORCING_ORDER = ("phis", "sgh30", "landfrac")
SCREAM_LEVEL_START = 8
SCREAM_LEVEL_END = 32
SCREAM_PLEVEL = 1


def _open_array(fs: s3fs.S3FileSystem, path: str) -> zarr.Array:
    return zarr.open_array(s3fs.S3Map(root=path, s3=fs, check=False), mode="r")


def _hrrr_run_roots(analysis_time: np.datetime64) -> tuple[str, str]:
    yyyymmdd = str(analysis_time.astype("datetime64[D]")).replace("-", "")
    hh = int(str(analysis_time.astype("datetime64[h]"))[-2:])
    prs_root = f"s3://hrrrzarr/prs/{yyyymmdd}/{yyyymmdd}_{hh:02d}z_anl.zarr"
    sfc_root = f"s3://hrrrzarr/sfc/{yyyymmdd}/{yyyymmdd}_{hh:02d}z_anl.zarr"
    return prs_root, sfc_root


def _hrrr_latlon(prs_root: str, fs: s3fs.S3FileSystem) -> tuple[np.ndarray, np.ndarray]:
    """Derive HRRR lat/lon from native projection coordinates."""
    x = _open_array(fs, f"{prs_root}/1000mb/TMP/projection_x_coordinate")[:].astype(
        "float64"
    )
    y = _open_array(fs, f"{prs_root}/1000mb/TMP/projection_y_coordinate")[:].astype(
        "float64"
    )
    xx, yy = np.meshgrid(x, y)
    # HRRR Lambert conformal parameters from public metadata/docs.
    crs = CRS.from_cf(
        {
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": (38.5, 38.5),
            "latitude_of_projection_origin": 38.5,
            "longitude_of_central_meridian": 262.5,
            "earth_radius": 6371229.0,
        }
    )
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return lat.astype("float32"), lon.astype("float32")


def _tile_slices(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    tile_size: int,
) -> tuple[slice, slice, tuple[int, int]]:
    dist = (lat - center_lat) ** 2 + (lon - center_lon) ** 2
    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
    h, w = lat.shape
    half = tile_size // 2
    y0 = int(np.clip(iy - half, 0, h - tile_size))
    x0 = int(np.clip(ix - half, 0, w - tile_size))
    return slice(y0, y0 + tile_size), slice(x0, x0 + tile_size), (iy, ix)


def _read_hrrr_var(
    fs: s3fs.S3FileSystem,
    root: str,
    level: str,
    var_name: str,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    path = f"{root}/{level}/{var_name}/{level}/{var_name}"
    arr = _open_array(fs, path)
    return np.asarray(arr[y_slice, x_slice], dtype=np.float32)


def fetch_hrrr_tile(
    analysis_time: np.datetime64,
    center_lat: float,
    center_lon: float,
    tile_size: int,
) -> dict[str, np.ndarray | np.datetime64 | str | tuple[int, int]]:
    fs = s3fs.S3FileSystem(anon=True)
    prs_root, sfc_root = _hrrr_run_roots(analysis_time)
    lat, lon = _hrrr_latlon(prs_root, fs)
    y_slice, x_slice, nearest_idx = _tile_slices(
        lat, lon, center_lat, center_lon, tile_size
    )
    lat_tile = lat[y_slice, x_slice]
    lon_tile = lon[y_slice, x_slice]

    def _read_pr(var_name: str) -> np.ndarray:
        return np.stack(
            [
                _read_hrrr_var(fs, prs_root, f"{level}mb", var_name, y_slice, x_slice)
                for level in HRRR_PR_LEVELS_HPA
            ],
            axis=0,
        )

    return {
        "analysis_time": analysis_time,
        "prs_root": prs_root,
        "sfc_root": sfc_root,
        "nearest_index": nearest_idx,
        "lat_tile": lat_tile.astype("float32"),
        "lon_tile": lon_tile.astype("float32"),
        "temp": _read_pr("TMP"),
        "u": _read_pr("UGRD"),
        "v": _read_pr("VGRD"),
        "z_mid": _read_pr("HGT"),
        "omega": _read_pr("VVEL"),
        "qv": _read_pr("SPFH"),
        "t2m": _read_hrrr_var(fs, sfc_root, "2m_above_ground", "TMP", y_slice, x_slice),
        "hgt_sfc": _read_hrrr_var(fs, sfc_root, "surface", "HGT", y_slice, x_slice),
        "ps": _read_hrrr_var(fs, sfc_root, "surface", "PRES", y_slice, x_slice),
        "landfrac": _read_hrrr_var(fs, sfc_root, "surface", "LAND", y_slice, x_slice),
    }


def _target_pressures_from_aux(
    vertical_coordinate_path: str,
    surface_pressure_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    with xr.open_dataset(vertical_coordinate_path) as ds_vert:
        hyam_all = ds_vert["hyam"].values.astype(np.float32)
        hybm_all = ds_vert["hybm"].values.astype(np.float32)
        p0_pa = float(ds_vert["P0"].values) if "P0" in ds_vert else 100000.0
    level_indices = np.arange(
        SCREAM_LEVEL_START, SCREAM_LEVEL_END, SCREAM_PLEVEL, dtype=np.int64
    )
    hyam = hyam_all[level_indices]
    hybm = hybm_all[level_indices]
    target_pressure = (
        hyam[:, None, None] * p0_pa + hybm[:, None, None] * surface_pressure_pa[None]
    ).astype(np.float32)
    return target_pressure, level_indices, p0_pa


def build_scream_inputs(
    hrrr: dict[str, np.ndarray | np.datetime64 | str | tuple[int, int]],
    in_variable_names: list[str],
    vertical_coordinate_path: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    ps = np.asarray(hrrr["ps"], dtype=np.float32)
    target_pressure, level_indices, p0_pa = _target_pressures_from_aux(
        vertical_coordinate_path, ps
    )

    src_pressure = (np.asarray(HRRR_PR_LEVELS_HPA, dtype=np.float32) * 100.0)[
        :, None, None
    ]

    def _v_interp(values_3d: np.ndarray) -> np.ndarray:
        src_vals = torch.from_numpy(values_3d.astype(np.float32))
        src_p = torch.from_numpy(np.broadcast_to(src_pressure, values_3d.shape).copy())
        tgt_p = torch.from_numpy(target_pressure)
        interp = log_pressure_interpolate(
            src_values=src_vals,
            src_pressure=src_p,
            target_pressure=tgt_p,
            axis=0,
        )
        return interp.detach().cpu().numpy()

    temp_on_scream = _v_interp(np.asarray(hrrr["temp"], dtype=np.float32))
    u_on_scream = _v_interp(np.asarray(hrrr["u"], dtype=np.float32))
    v_on_scream = _v_interp(np.asarray(hrrr["v"], dtype=np.float32))
    zmid_on_scream = _v_interp(np.asarray(hrrr["z_mid"], dtype=np.float32))
    omega_on_scream = _v_interp(np.asarray(hrrr["omega"], dtype=np.float32))
    qv_on_scream = _v_interp(np.asarray(hrrr["qv"], dtype=np.float32))

    theta_on_scream = potential_temperature_from_temperature(
        temperature_k=torch.from_numpy(temp_on_scream),
        pressure_pa=torch.from_numpy(target_pressure),
        p0_pa=float(p0_pa),
    ).detach().cpu().numpy()

    channels: dict[str, np.ndarray] = {}
    for i, lev in enumerate(level_indices):
        channels[f"PotentialTemperature_{lev}"] = theta_on_scream[i]
        channels[f"U_{lev}"] = u_on_scream[i]
        channels[f"V_{lev}"] = v_on_scream[i]
        channels[f"z_mid_{lev}"] = zmid_on_scream[i]
        channels[f"omega_{lev}"] = omega_on_scream[i]
        channels[f"qv_{lev}"] = qv_on_scream[i]
    channels["T_2m"] = np.asarray(hrrr["t2m"], dtype=np.float32)

    x_np = np.stack([channels[name] for name in in_variable_names], axis=0).astype(
        np.float32
    )[None, :, None]

    phis = np.asarray(hrrr["hgt_sfc"], dtype=np.float32) * GRAVITY
    sgh30 = np.zeros_like(phis, dtype=np.float32)
    landfrac = np.asarray(hrrr["landfrac"], dtype=np.float32)
    static_tile = np.stack([phis, sgh30, landfrac], axis=0).astype(np.float32)[:, None]

    stats = {
        "theta": (float(theta_on_scream.min()), float(theta_on_scream.max())),
        "qv": (float(qv_on_scream.min()), float(qv_on_scream.max())),
        "u": (float(u_on_scream.min()), float(u_on_scream.max())),
        "v": (float(v_on_scream.min()), float(v_on_scream.max())),
    }
    return x_np, static_tile, stats


def load_model_without_scream_zarr(
    checkpoint_path: str,
    bf16: bool = True,
) -> e2w.ScreamcastModel:
    """Load the released checkpoint while skipping SCREAM-zarr static forcing read."""
    orig_static = e2w._SCREAM_STATIC_FORCING
    try:
        # Skip static forcing load in from_checkpoint; static fields come from HRRR tile.
        e2w._SCREAM_STATIC_FORCING = frozenset()
        model = e2w.ScreamcastModel.from_checkpoint(checkpoint_path, bf16=bf16)
    finally:
        e2w._SCREAM_STATIC_FORCING = orig_static
    model._static_forcing_names = list(STATIC_FORCING_ORDER)
    return model


def configure_model_for_hrrr_tile(
    model: e2w.ScreamcastModel,
    lat_tile: np.ndarray,
    lon_tile: np.ndarray,
    static_tile: np.ndarray,
    tile_size: int,
) -> None:
    device = next(model.parameters()).device
    lat_t = torch.from_numpy(lat_tile).to(device=device, dtype=torch.float32)
    lon_t = torch.from_numpy(lon_tile).to(device=device, dtype=torch.float32)
    tile_latlon = torch.stack([lat_t, lon_t], dim=-1).unsqueeze(0)  # [1,H,W,2]
    static_t = torch.from_numpy(static_tile).to(device=device, dtype=torch.float32)

    # Seed source buffers so set_latlon can run without SCREAM static fields.
    model.register_buffer("latlon", tile_latlon, persistent=False)
    model.register_buffer("_src_latlon", tile_latlon, persistent=False)
    model.register_buffer("static_forcing", static_t, persistent=False)
    model.register_buffer("_src_static_forcing", static_t, persistent=False)
    model._n_faces = 1

    model.set_tile_size(tile_size)
    model.set_latlon(lat_deg=lat_t.unsqueeze(0), lon_deg=lon_t.unsqueeze(0))


def run_rollout(
    model: e2w.ScreamcastModel,
    x_np: np.ndarray,
    analysis_time: np.datetime64,
    n_steps: int,
) -> None:
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)

    coords = dict(model.input_coords())
    coords["batch"] = np.empty(1)
    coords["time"] = np.array([analysis_time])
    coords["face"] = np.array([0], dtype=np.int64)
    coords["x"] = np.arange(x_np.shape[-2], dtype=np.int64)
    coords["y"] = np.arange(x_np.shape[-1], dtype=np.int64)

    dt_minutes = model.dt / np.timedelta64(1, "s") / 60.0
    print(
        f"rollout setup: input_shape={tuple(x_np.shape)} steps={n_steps} dt_minutes={dt_minutes:.1f}"
    )
    with torch.inference_mode():
        iterator = model.create_iterator(x_tensor, coords)
        next(iterator)  # lead_time=0
        for step in range(1, n_steps + 1):
            out, out_coords = next(iterator)
            out_np = out.detach().cpu().numpy()
            has_nan = bool(np.isnan(out_np).any())
            print(
                f"step={step:02d} lead_hours={float(dt_minutes * step / 60.0):.2f} "
                f"min={float(out_np.min()):.5g} max={float(out_np.max()):.5g} nan={has_nan}"
            )
            if has_nan:
                raise RuntimeError("NaN detected in model output.")
    print("rollout completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("STRATA_CHECKPOINT", "checkpoints/strata_v1.pth"),
        help="Path to HF checkpoint file (default: checkpoints/strata_v1.pth or STRATA_CHECKPOINT).",
    )
    parser.add_argument(
        "--analysis-time",
        default="2026-07-01T12:00:00",
        help="HRRR analysis time in ISO format.",
    )
    parser.add_argument("--center-lat", type=float, default=39.0)
    parser.add_argument("--center-lon", type=float, default=-97.0)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU only.")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 setup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}\n"
            "Download from HF, e.g.:\n"
            "  hf download nvidia/strata checkpoints/strata_v1.pth --local-dir /mnt/nvme/strata_v1"
        )

    aux_root = Path(os.environ.get("AUX_DATA_ROOT", "data")).expanduser().resolve()
    latlon_path = aux_root / "latlon_ne1024pg2.nc"
    vertical_path = aux_root / "scream_vertical_coordinate.nc"
    if not latlon_path.exists():
        raise FileNotFoundError(f"missing required file: {latlon_path}")
    if not vertical_path.exists():
        raise FileNotFoundError(f"missing required file: {vertical_path}")

    analysis_time = np.datetime64(args.analysis_time)
    print(f"checkpoint={checkpoint_path}")
    print(f"AUX_DATA_ROOT={aux_root}")
    print(f"analysis_time={analysis_time}")

    hrrr = fetch_hrrr_tile(
        analysis_time=analysis_time,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        tile_size=args.tile_size,
    )
    print(
        f"hrrr roots: prs={hrrr['prs_root']} sfc={hrrr['sfc_root']} nearest={hrrr['nearest_index']}"
    )
    print(
        f"tile lat range=({float(np.min(hrrr['lat_tile'])):.2f}, {float(np.max(hrrr['lat_tile'])):.2f}) "
        f"lon range=({float(np.min(hrrr['lon_tile'])):.2f}, {float(np.max(hrrr['lon_tile'])):.2f})"
    )

    model = load_model_without_scream_zarr(
        checkpoint_path=str(checkpoint_path),
        bf16=not args.no_bf16,
    )
    if args.cpu:
        model = model.cpu()
    else:
        model = model.cuda()
    model.eval()

    input_vars = list(model.input_coords()["variable"])
    x_np, static_tile, stats = build_scream_inputs(
        hrrr=hrrr,
        in_variable_names=input_vars,
        vertical_coordinate_path=str(vertical_path),
    )
    print(f"input tensor shape={tuple(x_np.shape)}")
    for key, (vmin, vmax) in stats.items():
        print(f"{key} range=({vmin:.5g}, {vmax:.5g})")

    configure_model_for_hrrr_tile(
        model=model,
        lat_tile=np.asarray(hrrr["lat_tile"], dtype=np.float32),
        lon_tile=np.asarray(hrrr["lon_tile"], dtype=np.float32),
        static_tile=static_tile,
        tile_size=args.tile_size,
    )
    run_rollout(
        model=model,
        x_np=x_np,
        analysis_time=analysis_time,
        n_steps=args.steps,
    )


if __name__ == "__main__":
    main()
