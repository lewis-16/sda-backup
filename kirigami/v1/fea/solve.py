import numpy as np

UM_TO_M = 1e-6


def run_fea(mesh_data, force_N=1e-6, E_Pa=3e9, nu=0.34):
    import skfem as fem
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from scipy.sparse.linalg import spsolve

    points_3d = mesh_data["points_3d"]
    tets = mesh_data["tets"]
    fixed_ids = mesh_data["fixed_ids"]

    scale = UM_TO_M
    p = np.ascontiguousarray((points_3d * scale).T, dtype=np.float64)
    t = np.ascontiguousarray(tets.T, dtype=np.int64)
    mesh = fem.MeshTet(p, t)

    elem = fem.ElementVector(fem.ElementTetP1())
    basis = fem.CellBasis(mesh, elem)
    lam, mu = lame_parameters(E_Pa, nu)
    K = fem.asm(linear_elasticity(lam, mu), basis)

    fixed_dofs = np.ravel([[3 * i, 3 * i + 1, 3 * i + 2] for i in fixed_ids])
    free = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)

    z_top = 49e-6
    r_in2 = (49e-6) ** 2
    r_out2 = (86e-6) ** 2

    def load_facets(x):
        r2 = x[0] ** 2 + x[1] ** 2
        return (x[2] > z_top) & (r2 > r_in2) & (r2 < r_out2)

    load_facet_ids = mesh.facets_satisfying(load_facets, boundaries_only=True)
    shuttle_area_m2 = np.pi * ((85e-6) ** 2 - (50e-6) ** 2)
    traction = force_N / shuttle_area_m2 if shuttle_area_m2 > 0 else 0
    b = np.zeros(K.shape[0])
    if len(load_facet_ids) > 0:
        fbasis = fem.FacetBasis(mesh, elem, facets=load_facet_ids)

        @fem.LinearForm
        def load(v, w):
            return (-traction) * v[2]

        b = fem.asm(load, fbasis)

    u = np.zeros(K.shape[0])
    u[free] = spsolve(K[free][:, free], b[free])

    u_nodal = u.reshape(-1, 3)
    return _SkfemResult(u_nodal, mesh, basis)


def run_fea_sharingan(mesh_data, force_N=1e-6, centrifugal_omega2=0.0, density_kg_m3=1200.0, E_Pa=3e9, nu=0.34):
    import skfem as fem
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from scipy.sparse.linalg import spsolve

    points_3d = mesh_data["points_3d"]
    tets = mesh_data["tets"]
    fixed_ids = mesh_data["fixed_ids"]
    r_inner_load_um = mesh_data.get("r_inner_load_um", 350.0)
    thickness = 50.0
    if points_3d.size > 0:
        z_vals = points_3d[:, 2]
        thickness = float(np.max(z_vals) - np.min(z_vals))

    scale = UM_TO_M
    p = np.ascontiguousarray((points_3d * scale).T, dtype=np.float64)
    t = np.ascontiguousarray(tets.T, dtype=np.int64)
    mesh = fem.MeshTet(p, t)

    elem = fem.ElementVector(fem.ElementTetP1())
    basis = fem.CellBasis(mesh, elem)
    lam, mu = lame_parameters(E_Pa, nu)
    K = fem.asm(linear_elasticity(lam, mu), basis)

    fixed_dofs = np.ravel([[3 * i, 3 * i + 1, 3 * i + 2] for i in fixed_ids])
    free = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)

    z_top_m = (np.max(points_3d[:, 2]) - 0.5) * scale
    r_inner_m = r_inner_load_um * scale
    r_inner_tol_m = max(50e-6, 0.1 * r_inner_m)

    def load_facets(x):
        r2 = x[0] ** 2 + x[1] ** 2
        return (x[2] > z_top_m) & (r2 <= (r_inner_m + r_inner_tol_m) ** 2)

    load_facet_ids = mesh.facets_satisfying(load_facets, boundaries_only=True)
    b = np.zeros(K.shape[0])

    if len(load_facet_ids) > 0:
        fbasis = fem.FacetBasis(mesh, elem, facets=load_facet_ids)

        @fem.LinearForm
        def load_unit(v, w):
            return v[2]

        b_load = fem.asm(load_unit, fbasis)
        load_area_m2 = np.abs(np.sum(b_load))
        if load_area_m2 < 1e-20:
            load_area_m2 = np.pi * ((r_inner_m + r_inner_tol_m) ** 2 - max(0, r_inner_m - r_inner_tol_m) ** 2)
        traction = -force_N / load_area_m2 if load_area_m2 > 0 else 0

        @fem.LinearForm
        def load(v, w):
            return traction * v[2]

        b += fem.asm(load, fbasis)

    if centrifugal_omega2 > 0 and density_kg_m3 > 0:
        body_force_coeff = density_kg_m3 * centrifugal_omega2

        @fem.LinearForm
        def body(v, w):
            x = w.x
            return body_force_coeff * (x[0] * v[0] + x[1] * v[1])

        b += fem.asm(body, basis)

    u = np.zeros(K.shape[0])
    u[free] = spsolve(K[free][:, free], b[free])

    u_nodal = u.reshape(-1, 3)
    return _SkfemResult(u_nodal, mesh, basis)


class _SkfemResult:
    def __init__(self, u_nodal, mesh, basis):
        self.u_nodal = u_nodal
        self.mesh = mesh
        self.basis = basis
