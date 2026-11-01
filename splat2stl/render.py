"""Pure-PyTorch 3D Gaussian Splatting renderer (no CUDA extension to compile).

EWA projection of 3D covariances, alpha = opacity * exp(-1/2 d^T Sigma^-1 d), exact front-to-back
compositing per pixel (sort by pixel+depth, segmented cumulative log-transmittance). Same image as a
standard 3DGS viewer for degree-0 (view-independent) colors. Cameras use the COLMAP/OpenCV convention
(x right, y down, z forward)."""
import math, numpy as np, torch
from plyfile import PlyData

C0 = 0.28209479177387814


def load_gs(path, dev="cuda"):
    """Read a standard 3DGS .ply (x,y,z, f_dc_*, opacity, scale_*, rot_*). Only the DC color is used."""
    v = PlyData.read(path)["vertex"].data
    f = lambda *k: torch.tensor(np.stack([v[x] for x in k], 1).astype(np.float32), device=dev)
    xyz = f("x", "y", "z"); rgb = (0.5 + C0 * f("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1)
    op = torch.sigmoid(f("opacity")[:, 0]); sc = torch.exp(f("scale_0", "scale_1", "scale_2"))
    q = torch.nn.functional.normalize(f("rot_0", "rot_1", "rot_2", "rot_3"), dim=1)
    w, x, y, z = q.unbind(1)
    R = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1).view(-1, 3, 3)
    M = R * sc[:, None, :]
    return dict(xyz=xyz, rgb=rgb, op=op, cov=M @ M.transpose(1, 2))


def look_at(C, T, up):
    z = T - C; z /= np.linalg.norm(z)
    y = -up - np.dot(-up, z) * z; y /= np.linalg.norm(y)      # image 'down' = against world 'up'
    x = np.cross(y, z)
    Rw = np.stack([x, y, z]); return Rw, -Rw @ C               # world -> camera


def sphere_cams(n, center, radius, up, min_elev_deg=-60):
    """n cameras on a Fibonacci sphere around `center`, looking at it; views below `min_elev_deg` are dropped."""
    k = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * k / n); th = np.pi * (1 + 5 ** 0.5) * k
    d = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    elev = np.degrees(np.arcsin(np.clip(d @ up, -1, 1)))
    d = d[elev >= min_elev_deg]
    return [look_at(center + radius * di, center, up) for di in d]


@torch.no_grad()
def render(gs, Rw, t, W, H, fx, fy, cx, cy, bg=1.0, rmax=24, chunk=6_000_000):
    """Returns (rgb HxWx3 in [0,1], accumulated alpha HxW, expected depth HxW)."""
    dev = gs["xyz"].device; Rt = torch.tensor(Rw, dtype=torch.float32, device=dev); tt = torch.tensor(t, dtype=torch.float32, device=dev)
    pc = gs["xyz"] @ Rt.T + tt; z = pc[:, 2]; ok = z > 0.05
    pc, z = pc[ok], z[ok]; op, rgb, cov = gs["op"][ok], gs["rgb"][ok], gs["cov"][ok]
    u = fx * pc[:, 0] / z + cx; v = fy * pc[:, 1] / z + cy
    J = torch.zeros(len(z), 2, 3, device=dev)
    J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * pc[:, 0] / z ** 2; J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * pc[:, 1] / z ** 2
    T = J @ Rt
    c2 = T @ cov @ T.transpose(1, 2) + 0.3 * torch.eye(2, device=dev)
    a, b, c = c2[:, 0, 0], c2[:, 0, 1], c2[:, 1, 1]; det = a * c - b * b
    ia, ib, ic = c / det, -b / det, a / det
    lam = 0.5 * (a + c) + torch.sqrt(torch.clamp(0.25 * (a - c) ** 2 + b * b, min=0))
    r = torch.clamp(torch.ceil(3 * torch.sqrt(lam)), 1, rmax).long()
    inside = (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H)
    keys, al, cols, deps = [], [], [], []
    for rr in torch.unique(r[inside]).tolist():
        sel = torch.nonzero(inside & (r == rr))[:, 0]
        o = torch.arange(-rr, rr + 1, device=dev)
        dy, dx = torch.meshgrid(o, o, indexing="ij"); dx = dx.reshape(-1); dy = dy.reshape(-1)
        for ch in torch.split(sel, max(1, chunk // len(dx))):
            px = torch.floor(u[ch])[:, None].long() + dx; py = torch.floor(v[ch])[:, None].long() + dy
            ddx = px + 0.5 - u[ch][:, None]; ddy = py + 0.5 - v[ch][:, None]
            pw = -0.5 * (ia[ch][:, None] * ddx ** 2 + 2 * ib[ch][:, None] * ddx * ddy + ic[ch][:, None] * ddy ** 2)
            A = torch.clamp(op[ch][:, None] * torch.exp(pw), max=0.99)
            m = (A > 1 / 255) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            keys.append((py * W + px)[m]); al.append(A[m])
            cols.append(rgb[ch][:, None, :].expand(-1, len(dx), -1)[m]); deps.append(z[ch][:, None].expand(-1, len(dx))[m])
    key = torch.cat(keys); A = torch.cat(al); col = torch.cat(cols); dep = torch.cat(deps)
    dq = ((dep - dep.min()) / (dep.max() - dep.min() + 1e-9) * (2 ** 22 - 1)).long()
    order = torch.argsort(key * (2 ** 22) + dq); key, A, col, dep = key[order], A[order], col[order], dep[order]
    la = torch.log1p(-A); cs = torch.cumsum(la, 0)
    newseg = torch.ones_like(key, dtype=torch.bool); newseg[1:] = key[1:] != key[:-1]
    segid = torch.cumsum(newseg.long(), 0) - 1; start = torch.nonzero(newseg)[:, 0]
    base = (cs[start] - la[start])[segid]
    w = torch.exp(cs - la - base) * A                           # transmittance before layer * its alpha
    img = torch.zeros(W * H, 3, device=dev).index_add_(0, key, w[:, None] * col)
    acc = torch.zeros(W * H, device=dev).index_add_(0, key, w)
    dmap = torch.zeros(W * H, device=dev).index_add_(0, key, w * dep)
    img = img + (1 - acc[:, None]) * bg
    return img.view(H, W, 3).clamp(0, 1), acc.view(H, W), (dmap / acc.clamp(min=1e-6)).view(H, W)
