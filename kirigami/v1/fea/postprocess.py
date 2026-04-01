import numpy as np

UM_TO_M = 1e-6


def _lame_parameters(E_Pa, nu):
    lam = E_Pa * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E_Pa / (2 * (1 + nu))
    return lam, mu


def compute_von_mises_nodal(mesh_data, result, E_Pa, nu):
    points_3d = mesh_data["points_3d"] * UM_TO_M
    tets = mesh_data["tets"]
    u = result.u_nodal
    lam, mu = _lame_parameters(E_Pa, nu)
    n_nodes = len(points_3d)
    stress_cell = np.zeros(len(tets))
    for i, tet in enumerate(tets):
        p = points_3d[tet]
        u_tet = u[tet]
        v1, v2, v3 = p[1] - p[0], p[2] - p[0], p[3] - p[0]
        V = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        if V < 1e-20:
            continue
        n0 = np.cross(p[1] - p[2], p[1] - p[3])
        n1 = np.cross(p[2] - p[0], p[2] - p[3])
        n2 = np.cross(p[3] - p[0], p[3] - p[1])
        n3 = np.cross(p[0] - p[1], p[0] - p[2])
        grad_n = np.array([n0, n1, n2, n3]) / (6.0 * V)
        grad_u = (u_tet.T @ grad_n).T
        eps = 0.5 * (grad_u + grad_u.T)
        trace_eps = np.trace(eps)
        sigma = lam * trace_eps * np.eye(3) + 2 * mu * eps
        trace_sigma = np.trace(sigma)
        s_dev = sigma - (trace_sigma / 3.0) * np.eye(3)
        von_mises = np.sqrt(1.5 * np.sum(s_dev ** 2))
        stress_cell[i] = von_mises
    node_count = np.zeros(n_nodes)
    von_mises_nodal = np.zeros(n_nodes)
    for i, tet in enumerate(tets):
        for n in tet:
            node_count[n] += 1
            von_mises_nodal[n] += stress_cell[i]
    von_mises_nodal = np.where(node_count > 0, von_mises_nodal / node_count, 0.0)
    return von_mises_nodal


def extract_results(result):
    u_nodal = result.u_nodal
    u_z = u_nodal[:, 2]
    scale = 1e6
    return {
        "u": u_nodal,
        "u_z": u_z,
        "uz_min_um": float(np.min(u_z) * scale),
        "uz_max_um": float(np.max(u_z) * scale),
        "uz_range_um": float((np.max(u_z) - np.min(u_z)) * scale),
    }


def save_results_vtk(result, output_path, mesh_data=None, point_data_extra=None):
    pd = {"u": result.u_nodal}
    if point_data_extra:
        pd.update(point_data_extra)
    if mesh_data is not None:
        import meshio
        mesh = meshio.Mesh(
            mesh_data["points_3d"],
            [("tetra", mesh_data["tets"])],
            point_data=pd,
        )
        mesh.write(output_path, file_format="vtk")
    else:
        result.mesh.save(output_path, point_data=pd)
    return output_path
