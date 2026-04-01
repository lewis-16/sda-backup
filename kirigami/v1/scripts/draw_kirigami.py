import os
import numpy as np
import phidl as ph

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
import phidl.geometry as pg
import ezdxf
from shapely.geometry import Polygon
from shapely.ops import unary_union


def fillet_polygon(points, radius, quad_segs=16):
    if len(points) < 3 or radius <= 0:
        return np.array(points)
    poly = Polygon(points).buffer(0)
    if poly.is_empty:
        return np.array(points)
    filleted = poly.buffer(-radius, join_style=1, quad_segs=quad_segs).buffer(
        radius, join_style=1, quad_segs=quad_segs
    )
    if filleted.is_empty:
        return np.array(points)
    if hasattr(filleted, 'geoms') and len(filleted.geoms) > 0:
        geom = max(filleted.geoms, key=lambda g: g.area)
    else:
        geom = filleted
    coords = np.array(geom.exterior.coords[:-1])
    return coords


def generate_radial_positions(r_i, r_o, dr1, dr2, n, dr_min):
    positions = []
    r = r_i + dr1
    if r >= r_o:
        return positions
    positions.append(r)

    r_next = r + dr2
    if r_next >= r_o:
        return positions
    positions.append(r_next)

    dr_prev = dr2
    r_curr = r_next
    while True:
        dr_next = r_o * ((1 + dr_prev / r_o) ** n - 1)
        if dr_next < dr_min:
            dr_next = dr_min
        r_next = r_curr + dr_next
        if r_next >= r_o - dr_min:
            break
        positions.append(r_next)
        dr_prev = dr_next
        r_curr = r_next
    return positions


def add_cross_bridge(D, r_inner, width, center_hole_diameter, layer=(2, 0)):
    half_w = width / 2
    r_hole = center_hole_diameter / 2
    arm_0 = ph.Device()
    arm_0.add_ref(pg.rectangle(size=(r_inner, width), layer=layer)).move((0, -half_w))
    arm_90 = ph.Device()
    arm_90.add_ref(pg.rectangle(size=(width, r_inner), layer=layer)).move((-half_w, 0))
    arm_180 = ph.Device()
    arm_180.add_ref(pg.rectangle(size=(r_inner, width), layer=layer)).move((-r_inner, -half_w))
    arm_270 = ph.Device()
    arm_270.add_ref(pg.rectangle(size=(width, r_inner), layer=layer)).move((-half_w, -r_inner))
    cross_01 = pg.boolean(arm_0, arm_90, operation="A+B", layer=layer)
    cross_012 = pg.boolean(cross_01, arm_180, operation="A+B", layer=layer)
    cross = pg.boolean(cross_012, arm_270, operation="A+B", layer=layer)
    center_hole = pg.circle(radius=r_hole, layer=(0, 0))
    bridge = pg.boolean(cross, center_hole, operation="A-B", layer=layer)
    D.add_ref(bridge)


def add_kirigami_to_device(D, r_i, r_o, positions, N_theta, theta, offset=0, slit_width=0.05, fillet_radius=0, slit_base_angle=0, layer=1):
    theta_i = 2 * np.pi / (N_theta * (1 + theta))
    theta_a = theta * theta_i
    theta_i_deg = np.degrees(theta_i)
    base_rad = np.radians(slit_base_angle)

    for j, r in enumerate(positions):
        phase = offset * j
        for k in range(N_theta):
            slit_center = k * (theta_i + theta_a) + phase + base_rad
            start_angle = slit_center - theta_i / 2
            start_deg = np.degrees(start_angle)
            arc_dev = pg.arc(radius=r, width=slit_width, theta=theta_i_deg, start_angle=start_deg, layer=layer)
            raw = arc_dev.get_polygons(by_spec=False)
            p0 = raw[0]
            poly = np.array(p0.points) if hasattr(p0, 'points') else np.array(p0)
            if fillet_radius > 0:
                poly = fillet_polygon(poly, fillet_radius)
            D.add_polygon(poly, layer=layer)


def _poly_to_shapely(poly, buffer_zero=False):
    pts = np.array(poly) if not hasattr(poly, "points") else np.array(poly.points)
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    if buffer_zero:
        p = p.buffer(0)
    if p.is_empty:
        return None
    return p


def _geom_to_polygon_coords(geom):
    if geom.is_empty:
        return []
    out = []
    if geom.geom_type == "Polygon":
        if not geom.exterior.is_empty:
            out.append(np.array(geom.exterior.coords[:-1]))
        for interior in geom.interiors:
            if not interior.is_empty:
                out.append(np.array(interior.coords[:-1]))
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            if not g.exterior.is_empty:
                out.append(np.array(g.exterior.coords[:-1]))
            for interior in g.interiors:
                if not interior.is_empty:
                    out.append(np.array(interior.coords[:-1]))
    return out


def phidl_device_to_dxf(D, output_path, include_io_pad=True):
    """导出 DXF，包含 Kirigami_platte 图层，可选包含 io_pad"""
    polys = D.get_polygons(by_spec=True)
    outline_geoms = [_poly_to_shapely(p, buffer_zero=True) for p in polys.get((0, 0), [])]
    bridge_geoms = [_poly_to_shapely(p, buffer_zero=True) for p in polys.get((2, 0), [])]
    kirigami_geoms = [_poly_to_shapely(p, buffer_zero=False) for p in polys.get((1, 0), [])]
    outline_geoms = [g for g in outline_geoms if g is not None]
    bridge_geoms = [g for g in bridge_geoms if g is not None]
    kirigami_geoms = [g for g in kirigami_geoms if g is not None]
    outline_plus_bridge = unary_union(outline_geoms + bridge_geoms) if (outline_geoms or bridge_geoms) else Polygon()
    if hasattr(outline_plus_bridge, "make_valid"):
        outline_plus_bridge = outline_plus_bridge.make_valid()
    tol = 0.1
    kirigami_buffed = [g.buffer(tol) for g in kirigami_geoms if not g.is_empty]
    kirigami_union = unary_union(kirigami_buffed) if kirigami_buffed else Polygon()
    if hasattr(kirigami_union, "make_valid"):
        kirigami_union = kirigami_union.make_valid()
    material = outline_plus_bridge.difference(kirigami_union)
    if hasattr(material, "make_valid"):
        material = material.make_valid()
    coords_list = _geom_to_polygon_coords(material)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("Kirigami_platte", color=1)
    for coords in coords_list:
        if len(coords) < 3:
            continue
        points = [(float(p[0]), float(p[1])) for p in coords]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "Kirigami_platte"})
    if include_io_pad:
        doc.layers.add("io_pad", color=2)
        for poly in polys.get((3, 0), []):
            pts = np.array(poly) if not hasattr(poly, "points") else np.array(poly.points)
            if len(pts) < 3:
                continue
            points = [(float(p[0]), float(p[1])) for p in pts]
            msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "io_pad"})
    doc.saveas(output_path)


def main():
    scale = 0.1  # 除中心圆环外其余结构缩放比例
    r_o = 25000.0 * scale  # kirigami_plattle 外半径 (μm)，外侧圆盘
    r_i = 15000.0 * scale  # kirigami_plattle 中心孔半径 (μm)
    dr1 = 1500.0 * scale  # 第一个切缝距中心孔的距离 (μm)
    dr2 = 1500.0 * scale  # 第二个切缝与第一个切缝的间距 (μm)
    n = 1.0  # 径向分布指数 (≥1)，n=1 等距，n>1 外圈间距逐渐增大
    N_theta = 8  # 角向扇区数量
    theta_ratio = 0.3  # 角度比 θ = θ_a / θ_i，控制切缝宽度与间隔
    dr_min = 300.0 * scale  # 最小材料宽度 (μm)，制造约束，径向间距小于此值则停止
    theta_i = 2 * np.pi / (N_theta * (1 + theta_ratio))  # 单个切缝的角度宽度 (弧度)
    theta_a = theta_ratio * theta_i  # 相邻切缝间的间隔角度 (弧度)
    offset = (theta_i + theta_a) / 2  # 相邻圈切缝的角度偏移 (弧度)，1/2 周期使切缝交错
    slit_width = 100.0 * scale  # 切缝径向宽度 (μm)，即 pg.arc 的 width，模拟激光切割的切缝宽度
    fillet_radius = 25.0 * scale  # 切缝倒角半径 (μm)，0 表示不倒角
    slit_base_angle = 22.5  # 第一层切缝中心线的基准角度 (度)，0°=正X轴逆时针。N_theta=4 时 45 使中心分布于 45/135/225/315 度
    connector_width = 170.0  # 十字形 bridge 臂宽 (μm)
    center_hole_diameter = 100.0  # 中心孔直径 (μm)
    tab_length = 10385.0  # io_pad 长 (μm)，沿切线
    tab_width_above = 3940.0  # 圆盘上方露出的宽度 (μm)
    tab_width_below = 2000.0  # 向下延伸与圆盘重叠的宽度 (μm)，重叠部分将减去
    tab_width = tab_width_above + tab_width_below  # 总宽度 5940 (μm)

    D = ph.Device("kirigami")
    kirigami_plattle_outer = pg.circle(radius=r_o, layer=(0, 0))
    kirigami_plattle_inner = pg.circle(radius=r_i, layer=(0, 0))
    kirigami_plattle = pg.boolean(kirigami_plattle_outer, kirigami_plattle_inner, operation="A-B", layer=(0, 0))
    D.add_ref(kirigami_plattle)
    add_cross_bridge(D, r_i, connector_width, center_hole_diameter, layer=(2, 0))

    tab_rect_dev = ph.Device()
    tab_rect_ref = tab_rect_dev.add_ref(pg.rectangle(size=(tab_length, tab_width), layer=(3, 0)))
    tab_rect_ref.move((-tab_length / 2, r_o - tab_width_below))
    tab_final = pg.boolean(tab_rect_dev, kirigami_plattle_outer, operation="A-B", layer=(3, 0))
    D.add_ref(tab_final)

    positions = generate_radial_positions(r_i, r_o, dr1, dr2, n, dr_min)
    add_kirigami_to_device(D, r_i, r_o, positions, N_theta, theta_ratio, offset, slit_width, fillet_radius, slit_base_angle, layer=(1, 0))

    p1 = os.path.join(RESULTS_DIR, "kirigami_pattern.dxf")
    p2 = os.path.join(RESULTS_DIR, "kirigami_pattern_woio.dxf")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    phidl_device_to_dxf(D, p1, include_io_pad=True)
    phidl_device_to_dxf(D, p2, include_io_pad=False)
    print(f"径向切缝位置 (μm): {[round(p, 2) for p in positions]}")
    print(f"已保存至 {p1}、{p2}")


if __name__ == "__main__":
    main()
