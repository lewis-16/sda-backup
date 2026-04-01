#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fea.mesh import create_meshio_mesh_sharingan
from fea.solve import run_fea_sharingan
from fea.postprocess import extract_results, save_results_vtk, compute_von_mises_nodal


def show_mesh_3d(mesh_path):
    try:
        import pyvista as pv
    except ImportError:
        print("安装 pyvista 后可用交互查看: pip install pyvista")
        print("或手动打开: %s" % mesh_path)
        return
    grid = pv.read(mesh_path)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, opacity=0.9)
    plotter.add_title("Sharingan 3D 网格")
    plotter.show()


def show_deformed(vtk_path, deform_scale=1e6, title=None, scalars_name=None):
    try:
        import pyvista as pv
    except ImportError:
        print("安装 pyvista 后可用交互查看: pip install pyvista")
        return
    grid = pv.read(vtk_path)
    if "u" not in grid.point_data:
        print("VTK 中无位移场 u")
        return
    u = grid.point_data["u"]
    deformed = grid.copy()
    deformed.points = deformed.points + deform_scale * u
    if scalars_name and scalars_name in deformed.point_data:
        scalars = deformed.point_data[scalars_name]
        sbar = {"title": scalars_name}
    else:
        scalars = u[:, 2] if u.ndim > 1 else None
        sbar = {"title": "u_z (m)"}
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, opacity=0.3, color="lightgray")
    plotter.add_mesh(deformed, show_edges=True, opacity=0.9, scalars=scalars, scalar_bar_args=sbar)
    if title:
        plotter.add_title(title)
    plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Sharingan FEA")
    parser.add_argument("dxf", nargs="?", default=None, help="DXF file path")
    parser.add_argument("-f", "--force", type=float, default=1e-1, help="Gravity force on inner circle (N)")
    parser.add_argument("--centrifugal", type=float, default=50.0, help="Centrifugal omega^2 (rad^2/s^2) for case 2")
    parser.add_argument("--density", type=float, default=1200.0, help="Density (kg/m^3)")
    parser.add_argument("--no-show", action="store_true", help="Do not show deformed plot")
    parser.add_argument("--deform-scale", type=float, default=1e6, help="Deformation scale for visualization")
    parser.add_argument("--thickness", type=float, default=50, help="Thickness (um)")
    parser.add_argument("--max-area", type=float, default=2500, help="Max triangle area for 2D mesh")
    parser.add_argument("--r-outer", type=float, default=10000, help="Outer circle radius for fixed BC (um)")
    parser.add_argument("--r-inner-load", type=float, default=350, help="Inner radius for load (um); top faces with r<=this get gravity")
    parser.add_argument("--show-mesh", action="store_true", help="Only build mesh, show 3D and exit (no FEA)")
    parser.add_argument("--gravity-only", action="store_true", help="Only run Case 1 (gravity), compute stress, show deformation and von Mises")
    parser.add_argument("--show-gravity", action="store_true", help="Only show gravity deformation (load existing VTK, no FEA)")
    parser.add_argument("--E", type=float, default=3e9, help="Young modulus (Pa)")
    parser.add_argument("--nu", type=float, default=0.34, help="Poisson ratio")
    args = parser.parse_args()

    args.dxf = args.dxf or os.path.join(RESULTS_DIR, "sharingan.dxf")
    mesh_out = os.path.join(RESULTS_DIR, "sharingan_mesh.vtk")
    out_gravity = os.path.join(RESULTS_DIR, "sharingan_fea_gravity.vtk")

    if args.show_gravity:
        if not os.path.isfile(out_gravity):
            raise SystemExit("重力结果不存在，请先运行: python scripts/run_fea_sharingan.py --gravity-only")
        show_deformed(out_gravity, deform_scale=args.deform_scale, title="重力荷载下的形变")
        return

    if not os.path.isfile(args.dxf):
        raise SystemExit("DXF file not found: %s" % args.dxf)

    out_shaking = os.path.join(RESULTS_DIR, "sharingan_fea_shaking.vtk")

    mesh, mesh_data = create_meshio_mesh_sharingan(
        args.dxf,
        thickness=args.thickness,
        max_area=args.max_area,
        r_outer_um=args.r_outer,
        r_inner_load_um=args.r_inner_load,
        material_layer="substrate",
        output_path=mesh_out,
    )
    print("Mesh: %d nodes, %d tetrahedra" % (mesh_data["points_3d"].shape[0], mesh_data["tets"].shape[0]))
    print("Fixed nodes: %d, Load facets: %d" % (len(mesh_data["fixed_ids"]), len(mesh_data["load_faces"])))
    print("3D mesh saved: %s" % mesh_out)

    if args.show_mesh:
        show_mesh_3d(mesh_out)
        return

    print("\n--- Case 1: 仅类器官重力 ---")
    result1 = run_fea_sharingan(
        mesh_data,
        force_N=args.force,
        centrifugal_omega2=0.0,
        density_kg_m3=args.density,
        E_Pa=args.E,
        nu=args.nu,
    )
    stats1 = extract_results(result1)
    print("u_z: min=%.4f um, max=%.4f um, range=%.4f um" % (stats1["uz_min_um"], stats1["uz_max_um"], stats1["uz_range_um"]))
    von_mises = compute_von_mises_nodal(mesh_data, result1, args.E, args.nu)
    vm_min, vm_max = float(np.min(von_mises)), float(np.max(von_mises))
    print("von Mises 应力: min=%.2f Pa, max=%.2f Pa" % (vm_min, vm_max))
    save_results_vtk(result1, out_gravity, mesh_data=mesh_data, point_data_extra={"von_mises_Pa": von_mises})
    print("Saved: %s" % out_gravity)

    if args.gravity_only:
        if not args.no_show:
            show_deformed(out_gravity, deform_scale=args.deform_scale, title="重力荷载: 形变 (着色 u_z)", scalars_name=None)
            show_deformed(out_gravity, deform_scale=args.deform_scale, title="重力荷载: 形变 (着色 von Mises 应力)", scalars_name="von_mises_Pa")
        return

    print("\n--- Case 2: 重力 + 离心力（摇晃）---")
    result2 = run_fea_sharingan(
        mesh_data,
        force_N=args.force,
        centrifugal_omega2=args.centrifugal,
        density_kg_m3=args.density,
        E_Pa=args.E,
        nu=args.nu,
    )
    stats2 = extract_results(result2)
    print("u_z: min=%.4f um, max=%.4f um, range=%.4f um" % (stats2["uz_min_um"], stats2["uz_max_um"], stats2["uz_range_um"]))
    save_results_vtk(result2, out_shaking, mesh_data=mesh_data)
    print("Saved: %s" % out_shaking)

    if not args.no_show:
        show_deformed(out_gravity, deform_scale=args.deform_scale, title="Case 1: 仅重力")
        show_deformed(out_shaking, deform_scale=args.deform_scale, title="Case 2: 重力+离心力")


if __name__ == "__main__":
    main()
