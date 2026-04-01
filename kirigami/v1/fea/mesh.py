import numpy as np
from shapely.geometry import Polygon

try:
    import triangle
    HAS_TRIANGLE = True
except ImportError:
    HAS_TRIANGLE = False

import meshio


def _shapely_to_triangle_input(geom):
    if geom.is_empty:
        return None
    geom = geom.buffer(0)
    if geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    verts_list = []
    segs_list = []
    holes_list = []
    if geom.geom_type == "Polygon":
        geoms = [geom]
    elif geom.geom_type == "MultiPolygon":
        geoms = list(geom.geoms)
    else:
        return None
    all_verts = []
    seg_offset = 0
    for g in geoms:
        if g.is_empty or g.exterior is None or len(g.exterior.coords) < 3:
            continue
        ext = np.array(g.exterior.coords[:-1])
        n_ext = len(ext)
        base = len(all_verts)
        all_verts.extend(ext.tolist())
        for i in range(n_ext):
            segs_list.append([base + i, base + (i + 1) % n_ext])
        for interior in g.interiors:
            if len(interior.coords) < 3:
                continue
            int_pts = np.array(interior.coords[:-1])
            n_int = len(int_pts)
            base = len(all_verts)
            all_verts.extend(int_pts.tolist())
            for i in range(n_int):
                segs_list.append([base + i, base + (i + 1) % n_int])
            try:
                hole_pt = Polygon(int_pts).representative_point()
                holes_list.append([hole_pt.x, hole_pt.y])
            except Exception:
                c = np.mean(int_pts, axis=0)
                holes_list.append([float(c[0]), float(c[1])])
    if not all_verts:
        return None
    return {
        "vertices": np.array(all_verts, dtype=np.float64),
        "segments": np.array(segs_list, dtype=np.int32) if segs_list else None,
        "holes": np.array(holes_list, dtype=np.float64) if holes_list else None,
    }


def _triangulate_triangle(geom, max_area, quality):
    tri_input = _shapely_to_triangle_input(geom)
    if tri_input is None:
        return None, None
    opts = "p"
    if max_area is not None and max_area > 0:
        opts += f"a{max_area}"
    if quality:
        opts += "q"
    tri_out = triangle.triangulate(tri_input, opts)
    return np.array(tri_out["vertices"], dtype=np.float64), np.array(tri_out["triangles"], dtype=np.int32)


def _triangulate_pygmsh(geom, max_area):
    import pygmsh
    mesh_size = np.sqrt(max_area) if max_area and max_area > 0 else 100
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon" or geom.is_empty:
        return None, None
    ext = np.array(geom.exterior.coords[:-1]).tolist()
    holes = []
    for interior in geom.interiors:
        pts = np.array(interior.coords[:-1])
        if len(pts) < 3:
            continue
        hole_poly = [p.tolist() for p in pts]
        holes.append(hole_poly)
    with pygmsh.geo.Geometry() as geo:
        hole_loops = []
        for hpts in holes:
            hp = geo.add_polygon(hpts, mesh_size=mesh_size, make_surface=False)
            hole_loops.append(hp.curve_loop)
        poly = geo.add_polygon(ext, mesh_size=mesh_size, holes=hole_loops if hole_loops else None)
        mesh = geo.generate_mesh(dim=2, verbose=False)
    points = mesh.points[:, :2]
    for c in mesh.cells:
        if c.type == "triangle":
            return np.asarray(points, dtype=np.float64), np.array(c.data, dtype=np.int64)
    return None, None


def triangulate_polygon(geom, max_area=None, quality=True):
    if HAS_TRIANGLE:
        try:
            return _triangulate_triangle(geom, max_area, quality)
        except Exception:
            pass
    verts, tris = _triangulate_pygmsh(geom, max_area)
    if verts is not None:
        return verts, tris
    raise RuntimeError("Triangulation failed (tried triangle and pygmsh)")


def extrude_2d_to_3d(vertices_2d, triangles, thickness, z_bottom=0):
    n = len(vertices_2d)
    pts_bottom = np.column_stack([vertices_2d, np.full(n, z_bottom)])
    pts_top = np.column_stack([vertices_2d, np.full(n, z_bottom + thickness)])
    points_3d = np.vstack([pts_bottom, pts_top])
    tets = []
    for a, b, c in triangles:
        i0, i1, i2 = int(a), int(b), int(c)
        j0, j1, j2 = i0 + n, i1 + n, i2 + n
        tets.append([i0, i1, i2, j2])
        tets.append([i0, i1, j2, j1])
        tets.append([i0, j1, j2, j0])
    return points_3d, np.array(tets, dtype=np.int64)


def build_mesh_with_regions(vertices_2d, triangles, thickness, r_o, shuttle_r_in=50, shuttle_r_out=85, tab_extent_y=None, z_bottom=0):
    points_3d, tets = extrude_2d_to_3d(vertices_2d, triangles, thickness, z_bottom)
    n = len(vertices_2d)
    r = np.sqrt(points_3d[:, 0] ** 2 + points_3d[:, 1] ** 2)
    z = points_3d[:, 2]
    tol_r = max(50, 0.02 * r_o)
    tol_z = thickness * 0.1
    if tab_extent_y is None:
        tab_extent_y = r_o + 3500
    surface_faces = _get_tet_surface_faces(tets, n)
    fixed_nodes = set()
    load_faces = []
    for face in surface_faces:
        nodes = list(face)
        xc = np.mean([points_3d[n, 0] for n in nodes])
        yc = np.mean([points_3d[n, 1] for n in nodes])
        zc = np.mean([points_3d[n, 2] for n in nodes])
        rc = np.sqrt(xc**2 + yc**2)
        on_top = zc >= z_bottom + thickness - tol_z
        on_bottom = zc <= z_bottom + tol_z
        on_side = not on_top and not on_bottom
        if on_top and shuttle_r_in - 5 <= rc <= shuttle_r_out + 5:
            load_faces.append(face)
        if on_side and rc >= r_o - tol_r:
            fixed_nodes.update(nodes)
        if on_top and rc >= r_o - tol_r:
            fixed_nodes.update(nodes)
        if yc >= tab_extent_y - tol_r and rc <= r_o + tol_r * 2:
            fixed_nodes.update(nodes)
    fixed_ids = np.array(sorted(fixed_nodes), dtype=np.int64)
    return points_3d, tets, fixed_ids, load_faces


def _get_tet_surface_faces(tets, n_bottom):
    n = n_bottom * 2
    face_to_cell = {}
    tet_faces = [
        (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
    ]
    for i, tet in enumerate(tets):
        for a, b, c in tet_faces:
            fa = tuple(sorted([int(tet[a]), int(tet[b]), int(tet[c])]))
            if fa in face_to_cell:
                del face_to_cell[fa]
            else:
                face_to_cell[fa] = (i, fa)
    return [f for _, f in face_to_cell.values()]


def build_mesh_with_regions_sharingan(vertices_2d, triangles, thickness, r_outer_um, r_inner_load_um, z_bottom=0):
    points_3d, tets = extrude_2d_to_3d(vertices_2d, triangles, thickness, z_bottom)
    n = len(vertices_2d)
    tol_r = max(100, 0.02 * r_outer_um)
    tol_z = thickness * 0.1
    surface_faces = _get_tet_surface_faces(tets, n)
    fixed_nodes = set()
    load_faces = []
    for face in surface_faces:
        nodes = list(face)
        xc = np.mean([points_3d[n, 0] for n in nodes])
        yc = np.mean([points_3d[n, 1] for n in nodes])
        zc = np.mean([points_3d[n, 2] for n in nodes])
        rc = np.sqrt(xc**2 + yc**2)
        on_top = zc >= z_bottom + thickness - tol_z
        if on_top and rc <= r_inner_load_um + tol_r:
            load_faces.append(face)
        if rc >= r_outer_um - tol_r:
            fixed_nodes.update(nodes)
    fixed_ids = np.array(sorted(fixed_nodes), dtype=np.int64)
    return points_3d, tets, fixed_ids, load_faces


def create_meshio_mesh_sharingan(dxf_path, thickness=50, max_area=2500, r_outer_um=10000, r_inner_load_um=350, material_layer="substrate", output_path=None):
    from .geometry import dxf_layers_to_material
    material = dxf_layers_to_material(dxf_path, material_layer=material_layer)
    if material is None:
        raise ValueError("Failed to extract material from DXF (layer=%s)" % material_layer)
    vertices_2d, triangles = triangulate_polygon(material, max_area=max_area)
    if vertices_2d is None:
        raise ValueError("Triangulation failed")
    points_3d, tets, fixed_ids, load_faces = build_mesh_with_regions_sharingan(
        vertices_2d, triangles, thickness, r_outer_um, r_inner_load_um
    )
    mesh = meshio.Mesh(
        points_3d,
        [("tetra", tets)],
        point_data={},
        cell_data={},
    )
    if output_path:
        mesh.write(output_path, file_format="vtk")
    return mesh, {
        "vertices_2d": vertices_2d,
        "triangles": triangles,
        "points_3d": points_3d,
        "tets": tets,
        "fixed_ids": fixed_ids,
        "load_faces": load_faces,
        "r_outer_um": r_outer_um,
        "r_inner_load_um": r_inner_load_um,
    }


def create_meshio_mesh(dxf_path, thickness=50, max_area=2500, r_o=2500, shuttle_r_in=50, shuttle_r_out=85, output_path=None):
    from .geometry import dxf_layers_to_material
    material = dxf_layers_to_material(dxf_path)
    if material is None:
        raise ValueError("Failed to extract material from DXF")
    vertices_2d, triangles = triangulate_polygon(material, max_area=max_area)
    if vertices_2d is None:
        raise ValueError("Triangulation failed")
    points_3d, tets, fixed_ids, load_faces = build_mesh_with_regions(
        vertices_2d, triangles, thickness, r_o, shuttle_r_in, shuttle_r_out
    )
    mesh = meshio.Mesh(
        points_3d,
        [("tetra", tets)],
        point_data={},
        cell_data={},
    )
    mesh.cell_sets = {"Omega": [np.arange(len(tets))]}
    mesh.point_sets = {"Fixed": fixed_ids}
    mesh.face_sets = {"LoadSurface": load_faces}
    if output_path:
        mesh.write(output_path, file_format="vtk")
    return mesh, {
        "vertices_2d": vertices_2d,
        "triangles": triangles,
        "points_3d": points_3d,
        "tets": tets,
        "fixed_ids": fixed_ids,
        "load_faces": load_faces,
    }
