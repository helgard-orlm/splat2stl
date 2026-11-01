"""3DGS splat -> watertight printable mesh by silhouette carving (visual hull).

Why silhouettes and not depth: image-to-3D generators (Idol, TripoSplat, ...) often build parts of an
object out of many nearly transparent Gaussians ("fog"). The viewer composites them into a smooth,
opaque-looking surface, but there is no well-defined depth inside the fog, so any depth/threshold-based
surface comes out pitted, bumpy or grainy. The *outline* of the object, however, is well defined: the
accumulated opacity through the whole object is high. So we render the splat from many directions,
take the outline (accumulated alpha > 0.5, the same thing you see in the viewer), and carve a voxel
block: everything that falls outside the outline in any view is removed. This is the classic visual
hull (shape from silhouette); concavities that are hidden in every silhouette are filled.
"""
import math
import numpy as np
import torch
import trimesh
from scipy import ndimage
from skimage import measure

from .render import load_gs, sphere_cams, render

UP_AXES = {"-y": (0, -1, 0), "y": (0, 1, 0), "z": (0, 0, 1), "-z": (0, 0, -1), "x": (1, 0, 0), "-x": (-1, 0, 0)}


def splat_to_mesh(ply, out_prefix, height_mm=80.0, up="-y", views=140, res=256, img=512, smooth=1,
                  alpha_thr=0.5, min_elev=-60.0, device="cuda", chunk=6_000_000, write_ply=False, verbose=True):
    """Convert a 3DGS .ply into `<out_prefix>.stl` (geometry) and `<out_prefix>_color.obj` (vertex colors).

    height_mm  -- final size along the up axis (the mesh is exported Z-up, in millimetres)
    up         -- which axis is "up" in the splat ("-y" for Idol/TripoSplat/most 3DGS generators)
    views      -- number of render directions on a sphere (views below `min_elev` degrees are skipped)
    res        -- voxel grid resolution along the longest side
    img        -- render resolution (square)
    smooth     -- 0..5, blur of the voxel solid + Taubin smoothing iterations
    alpha_thr  -- accumulated opacity that counts as "object" in a silhouette
    Returns a dict with mesh statistics.
    """
    upv = np.array(UP_AXES[up], float)
    gs = load_gs(ply, device)
    xyz = gs["xyz"].cpu().numpy(); op = gs["op"].cpu().numpy()
    sel = xyz[op > 0.1] if (op > 0.1).sum() > 100 else xyz
    lo_p, hi_p = np.percentile(sel, 0.5, 0), np.percentile(sel, 99.5, 0)
    ctr = (lo_p + hi_p) / 2
    half = float((hi_p - lo_p).max()) / 2 * 1.25 + 1e-6          # carving cube, with margin
    W = H = img; fov = math.radians(30); f = W / 2 / math.tan(fov / 2)
    radius = half * 1.15 / math.sin(fov / 2)                       # whole cube fits in the frame
    cams = sphere_cams(views, ctr, radius, upv, min_elev_deg=min_elev)
    lo = ctr - half; vox = 2 * half / res
    g = (np.stack(np.meshgrid(*[np.arange(res)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5).astype(np.float32)
    gt = torch.tensor(g * vox + lo.astype(np.float32), device=device)
    outside = torch.zeros(len(g), dtype=torch.bool, device=device)
    shots = []
    for Rw, t in cams:
        im, acc, _ = render(gs, Rw, t, W, H, f, f, W / 2, H / 2, chunk=chunk)
        mask = acc > alpha_thr
        Rt = torch.tensor(Rw, dtype=torch.float32, device=device); tt = torch.tensor(t, dtype=torch.float32, device=device)
        pc = gt @ Rt.T + tt
        u = torch.floor(f * pc[:, 0] / pc[:, 2] + W / 2).long(); v = torch.floor(f * pc[:, 1] / pc[:, 2] + H / 2).long()
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        outside |= ~inb | ~mask[v.clamp(0, H - 1), u.clamp(0, W - 1)]
        shots.append((Rw, t, (im.cpu().numpy() * 255).astype(np.uint8)))
    del gs, gt
    if device == "cuda":
        torch.cuda.empty_cache()
    occ = (~outside).cpu().numpy().reshape(res, res, res)
    occ = ndimage.binary_fill_holes(occ)
    lab, n = ndimage.label(occ)
    if n == 0:
        raise ValueError("empty volume: check `up` and that the .ply is a single object")
    sizes = ndimage.sum(occ, lab, range(1, n + 1)); occ = lab == (1 + int(np.argmax(sizes)))
    sm = max(0, int(smooth))
    field = np.pad(ndimage.gaussian_filter(occ.astype(np.float32), 0.8 + 0.35 * sm), 1)
    verts, faces, _, _ = measure.marching_cubes(field, 0.5)
    verts = (verts - 1 + 0.5) * vox + lo
    mesh = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    mesh.update_faces(mesh.nondegenerate_faces()); mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    trimesh.smoothing.filter_taubin(mesh, iterations=6 + 10 * sm)
    # vertex color from the most frontal render
    V, N = mesh.vertices, mesh.vertex_normals
    best = np.full(len(V), -2.0); col = np.full((len(V), 3), 255, np.uint8)
    for Rw, t, im in shots:
        C = -Rw.T @ t; dv = C - V; dv /= np.linalg.norm(dv, axis=1, keepdims=True)
        s = (N * dv).sum(1); upd = s > best
        if upd.any():
            pc = V[upd] @ Rw.T + t
            u = np.clip((f * pc[:, 0] / pc[:, 2] + W / 2).astype(int), 0, W - 1)
            v = np.clip((f * pc[:, 1] / pc[:, 2] + H / 2).astype(int), 0, H - 1)
            col[upd] = im[v, u]; best[upd] = s[upd]
    # export Z-up, millimetres, standing on Z=0, centred in XY
    zax = upv; xax = np.cross([0, 0, 1.0] if abs(zax[2]) < 0.9 else [1.0, 0, 0], zax); xax /= np.linalg.norm(xax)
    yax = np.cross(zax, xax)
    P = mesh.vertices @ np.stack([xax, yax, zax]).T
    P -= P.min(0); P *= height_mm / P[:, 2].max(); P[:, :2] -= P[:, :2].max(0) / 2
    mesh = trimesh.Trimesh(P, mesh.faces, vertex_colors=np.c_[col, np.full(len(col), 255, np.uint8)], process=False)
    trimesh.repair.fix_normals(mesh)
    tunnels = int(sum(max(0, (2 - c.euler_number) // 2) for c in mesh.split(only_watertight=False)))
    info = {"vertices": len(mesh.vertices), "faces": len(mesh.faces), "watertight": bool(mesh.is_watertight),
            "tunnels": tunnels, "volume_cm3": round(float(mesh.volume) / 1000, 1) if mesh.is_watertight else None,
            "size_mm": [round(float(x), 1) for x in mesh.extents], "views": len(cams),
            "fog_fraction": round(float(np.mean(op < 0.1)), 2)}
    mesh.export(out_prefix + ".stl"); mesh.export(out_prefix + "_color.obj")
    if write_ply:
        mesh.export(out_prefix + "_color.ply")
    return info
