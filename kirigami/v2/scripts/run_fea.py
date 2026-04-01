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

from fea.mesh import create_meshio_mesh
from fea.solve import run_fea as _run_fea
from fea.postprocess import extract_results, save_results_vtk, compute_von_mises_nodal


def show_mesh_3d(mesh_path):
    try:
        import pyvista as pv
    except ImportError:
        print(f"安装 pyvista 后可用交互查看: pip install pyvista")
        print(f"或手动打开: {mesh_path}")
        return
    grid = pv.read(mesh_path)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, opacity=0.9)
    plotter.show()


def show_deformed(vtk_path, deform_scale=1e6):
    try:
        import pyvista as pv
    except ImportError:
        print("安装 pyvista 后可用交互查看: pip install pyvista")
        return
    grid = pv.read(vtk_path)
    if "u" in grid.point_data:
        u = grid.point_data["u"]
        deformed = grid.copy()
        deformed.points = deformed.points + deform_scale * u
        u_z = u[:, 2] if u.ndim > 1 else None
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True, opacity=0.3, color="lightgray")
        plotter.add_mesh(deformed, show_edges=True, opacity=0.9, scalars=u_z, scalar_bar_args={"title": "u_z (m)"})
        plotter.add_title(f"形变放大 {deform_scale:.0e} 倍")
        plotter.show()
    else:
        print("VTK 中无位移场 u，请先运行 FEA")


def main():
    parser = argparse.ArgumentParser(description="Kirigami FEA")
    parser.add_argument("dxf", nargs="?", default=None, help="DXF file path")
    parser.add_argument("-f", "--force", type=float, default=1e-6, help="Central force (N)")
    parser.add_argument("-o", "--output", type=str, default=None, help="VTK output path")
    parser.add_argument("--mesh-out", type=str, default=None, help="Save mesh VTK path")
    parser.add_argument("--show-mesh", action="store_true", help="Show 3D extruded mesh and exit without FEA")
    parser.add_argument("--show-deformed", action="store_true", help="Show deformed mesh after FEA")
    parser.add_argument("--deform-scale", type=float, default=1e6, help="Deformation scale factor for visualization")
    parser.add_argument("--thickness", type=float, default=50, help="Thickness (um)")
    parser.add_argument("--max-area", type=float, default=2500, help="Max triangle area for 2D mesh")
    parser.add_argument("--E", type=float, default=3e9, help="Young modulus (Pa)")
    parser.add_argument("--nu", type=float, default=0.34, help="Poisson ratio")
    parser.add_argument("--yield-strength", type=float, default=None, help="Material yield/ultimate strength (Pa). If set, find max deformation before failure")
    parser.add_argument("--force-max", type=float, default=1e-3, help="Max force for strength scan (N)")
    parser.add_argument("--force-steps", type=int, default=20, help="Number of force steps for strength scan")
    parser.add_argument("--stress-percentile", type=float, default=95.0, help="Use percentile stress instead of max stress for failure criterion (0-100)")
    args = parser.parse_args()

    args.dxf = args.dxf or os.path.join(RESULTS_DIR, "kirigami_pattern_woio.dxf")
    args.output = args.output or os.path.join(RESULTS_DIR, "kirigami_fea.vtk")
    mesh_out_default = args.output.replace(".vtk", "_mesh.vtk") if args.output.endswith(".vtk") else os.path.join(RESULTS_DIR, "kirigami_mesh.vtk")
    args.mesh_out = args.mesh_out or mesh_out_default

    if not os.path.isfile(args.dxf):
        raise SystemExit(f"DXF file not found: {args.dxf}")

    mesh_out_path = args.mesh_out
    mesh, mesh_data = create_meshio_mesh(
        args.dxf,
        thickness=args.thickness,
        max_area=args.max_area,
        output_path=mesh_out_path,
    )
    print(f"Mesh: {mesh_data['points_3d'].shape[0]} nodes, {mesh_data['tets'].shape[0]} tetrahedra")
    print(f"Fixed nodes: {len(mesh_data['fixed_ids'])}, Load facets: {len(mesh_data['load_faces'])}")
    print(f"3D mesh saved: {mesh_out_path}")

    if args.show_mesh:
        show_mesh_3d(mesh_out_path)
        return

    if args.yield_strength is not None and args.yield_strength > 0:
        print(f"\n扫描载荷以找到最大安全形变（材料强度: {args.yield_strength/1e6:.1f} MPa）")
        print(f"载荷范围: {args.force:.2e} - {args.force_max:.2e} N, {args.force_steps} 步")
        force_values = np.linspace(args.force, args.force_max, args.force_steps)
        max_stress_values = []
        max_deform_values = []
        critical_force = None
        critical_result = None
        
        for i, f in enumerate(force_values):
            result = _run_fea(mesh_data, force_N=f, E_Pa=args.E, nu=args.nu)
            von_mises = compute_von_mises_nodal(mesh_data, result, args.E, args.nu)
            max_stress = float(np.max(von_mises))
            percentile_stress = float(np.percentile(von_mises, args.stress_percentile))
            effective_stress = percentile_stress if args.stress_percentile < 100 else max_stress
            stats = extract_results(result)
            max_deform = abs(stats['uz_max_um'])
            max_stress_values.append(max_stress)
            max_deform_values.append(max_deform)
            
            if i % max(1, args.force_steps // 10) == 0 or effective_stress >= args.yield_strength:
                print(f"  载荷 {f:.2e} N: {args.stress_percentile:.0f}%分位应力 {percentile_stress/1e6:.2f} MPa, 最大应力 {max_stress/1e6:.2f} MPa, 最大形变 {max_deform:.4f} um")
            
            if effective_stress >= args.yield_strength:
                critical_force = f
                critical_result = result
                print(f"\n达到材料强度！临界载荷: {critical_force:.2e} N")
                print(f"  {args.stress_percentile:.0f}%分位应力: {percentile_stress/1e6:.2f} MPa (目标: {args.yield_strength/1e6:.2f} MPa)")
                print(f"  最大应力: {max_stress/1e6:.2f} MPa")
                print(f"  最大形变: {max_deform:.4f} um")
                break
        
        if critical_force is None:
            final_result = _run_fea(mesh_data, force_N=args.force_max, E_Pa=args.E, nu=args.nu)
            final_von_mises = compute_von_mises_nodal(mesh_data, final_result, args.E, args.nu)
            final_percentile = float(np.percentile(final_von_mises, args.stress_percentile))
            print(f"\n在最大载荷 {args.force_max:.2e} N 下未达到材料强度")
            print(f"  {args.stress_percentile:.0f}%分位应力: {final_percentile/1e6:.2f} MPa (目标: {args.yield_strength/1e6:.2f} MPa)")
            print(f"  最大应力: {max_stress_values[-1]/1e6:.2f} MPa")
            print(f"  最大形变: {max_deform_values[-1]:.2f} um")
            critical_result = final_result
            critical_force = args.force_max
        
        result = critical_result
        stats = extract_results(result)
        von_mises = compute_von_mises_nodal(mesh_data, result, args.E, args.nu)
        percentile_stress = float(np.percentile(von_mises, args.stress_percentile))
        print(f"\n最终结果（载荷 {critical_force:.2e} N）:")
        print(f"u_z: min={stats['uz_min_um']:.4f} um, max={stats['uz_max_um']:.4f} um, range={stats['uz_range_um']:.4f} um")
        print(f"von Mises 应力: min={np.min(von_mises)/1e6:.2f} MPa, {args.stress_percentile:.0f}%分位={percentile_stress/1e6:.2f} MPa, max={np.max(von_mises)/1e6:.2f} MPa")
        save_results_vtk(result, args.output, mesh_data=mesh_data, point_data_extra={"von_mises_Pa": von_mises})
    else:
        result = _run_fea(mesh_data, force_N=args.force, E_Pa=args.E, nu=args.nu)
        stats = extract_results(result)
        print(f"u_z: min={stats['uz_min_um']:.4f} um, max={stats['uz_max_um']:.4f} um, range={stats['uz_range_um']:.4f} um")
        save_results_vtk(result, args.output, mesh_data=mesh_data)
    print(f"Saved: {args.output}")

    if args.show_deformed:
        show_deformed(args.output, deform_scale=args.deform_scale)


if __name__ == "__main__":
    main()
