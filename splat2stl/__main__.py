import argparse, json, time
from .hull import splat_to_mesh, UP_AXES


def main():
    ap = argparse.ArgumentParser(prog="python -m splat2stl",
                                 description="3DGS .ply -> watertight STL (+ vertex-colored OBJ) by silhouette carving")
    ap.add_argument("ply", help="input 3D Gaussian Splat .ply (single object)")
    ap.add_argument("out", help="output prefix: writes <out>.stl and <out>_color.obj")
    ap.add_argument("--height", type=float, default=80.0, help="size along the up axis, mm (default 80)")
    ap.add_argument("--up", choices=list(UP_AXES), default="-y", help="up axis of the splat (default -y)")
    ap.add_argument("--views", type=int, default=140, help="render directions (default 140)")
    ap.add_argument("--res", type=int, default=256, help="voxel grid resolution (default 256)")
    ap.add_argument("--img", type=int, default=512, help="render resolution (default 512)")
    ap.add_argument("--smooth", type=int, default=1, help="0..5 surface smoothing (default 1)")
    ap.add_argument("--alpha", type=float, default=0.5, help="silhouette threshold on accumulated opacity (default 0.5)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (cpu is slow)")
    ap.add_argument("--ply-out", action="store_true", help="also write <out>_color.ply")
    a = ap.parse_args()
    t = time.time()
    info = splat_to_mesh(a.ply, a.out, height_mm=a.height, up=a.up, views=a.views, res=a.res, img=a.img,
                         smooth=a.smooth, alpha_thr=a.alpha, device=a.device, write_ply=a.ply_out)
    info["seconds"] = round(time.time() - t, 1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
