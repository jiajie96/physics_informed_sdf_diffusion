import numpy as np
import point_cloud_utils as pcu
import torch
#from pytorch3d.ops import cubify, sample_points_from_meshes
#from pytorch3d.structures import Meshes
from trimesh import Trimesh
from skimage import measure


def mc_from_psr(psr_grid, pytorchify=False, real_scale=False, zero_level=0):
    """
    Run marching cubes from PSR grid
    from Shape as Points
    """
    batch_size = psr_grid.shape[0]
    s = psr_grid.shape[-1]  # size of psr_grid
    psr_grid_numpy = psr_grid.squeeze().detach().cpu().numpy()

    if batch_size > 1:
        verts, faces, normals = [], [], []
        for i in range(batch_size):
            verts_cur, faces_cur, normals_cur, values = measure.marching_cubes(psr_grid_numpy[i], level=0)
            verts.append(verts_cur)
            faces.append(faces_cur)
            normals.append(normals_cur)
        verts = np.stack(verts, axis=0)
        faces = np.stack(faces, axis=0)
        normals = np.stack(normals, axis=0)
    else:
        try:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy, level=zero_level)
        except:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy)
    if real_scale:
        verts = verts / (s - 1)  # scale to range [0, 1]
    else:
        verts = verts / s  # scale to range [0, 1)

    if pytorchify:
        device = psr_grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

    return verts, faces, normals


def make_pointclouds_grid(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), numpy or Tensor
    - return:
        - pts: N (3 or 6)
    """
    if isinstance(pts[0], torch.Tensor):
        return _make_pointclouds_grid_torch(pts, min_v, max_v, padding, nrows)
    elif isinstance(pts[0], np.ndarray):
        return _make_pointclouds_grid_numpy(pts, min_v, max_v, padding, nrows)
    else:
        raise TypeError


def _make_pointclouds_grid_numpy(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), numpy
    - return:
        - pts: N (3 or 6)
    """
    dist = max_v - min_v
    out_pts = []
    for i in range(len(pts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = np.array([[off_x, off_y, *((0,) * (pts[0].shape[-1] - 2))]])
        out_pts.append(pts[i] + offset)
    pts = np.concatenate(out_pts, 0)  # N (3 or 6)
    return pts


@torch.no_grad()
def _make_pointclouds_grid_torch(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), Tensor
    - return:
        - pts: N (3 or 6)
    """
    dist = max_v - min_v
    out_pts = []
    for i in range(len(pts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = pts[i].new_tensor([[off_x, off_y, *((0,) * (pts[0].shape[-1] - 2))]])
        out_pts.append(pts[i] + offset)
    pts = torch.cat(out_pts, 0)  # N (3 or 6)
    return pts


def make_meshes_grid(verts, faces, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - verts: list of (n (3 ~)), numpy
        - faces: list of n 3, numpy, int
    - return:
        - verts: n (3 ~)
        - faces: n 3
    """
    assert len(verts) == len(faces)

    dist = max_v - min_v
    face_offset = 0
    out_verts, out_faces = [], []
    for i in range(len(verts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = np.array([[off_x, off_y, *((0,) * (verts[0].shape[-1] - 2))]])
        out_verts.append(verts[i] + offset)
        out_faces.append(faces[i] + face_offset)
        face_offset += verts[i].shape[0]
    verts = np.concatenate(out_verts)  # N (3 or 6)
    faces = np.concatenate(out_faces)  # N 3
    return verts, faces


def random_color(verts):
    """
    - input:
        - verts: n 3
    """
    color = np.random.random(1, 3)
    color = np.repeat(color, verts.shape[0], 0)  # n 3
    return np.concatenate([verts, color], -1)  # n 6


def sdfs_to_meshes(psrs, safe=False, lib="trimesh"):
    """
    - input:
        - psrs: b 1 r r r
    - return:
        - meshes
    """
    mvs, mfs, mns = [], [], []
    for psr in psrs:
        if safe:
            try:
                mv, mf, mn = mc_from_psr(psr, pytorchify=True)
            except:
                mv = psrs.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                mf = psrs.new_tensor([[0, 1, 2]], dtype=torch.long)
                mn = psrs.new_tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        else:
            mv, mf, mn = mc_from_psr(psr, pytorchify=True)
        mvs.append(mv)
        mfs.append(mf)
        mns.append(mn)
    if lib == "pytorch3d":
        #meshes = Meshes(mvs, mfs, verts_normals=mns)
        pass # TODO
    if lib == "trimesh":
        meshes = list()
        for mv, mf, mn in zip(mvs, mfs, mns):
            meshes.append(Trimesh(vertices=mv.numpy(force=True), faces=mf.numpy(force=True), vertex_normals=mn.numpy(force=True)))
    return meshes


def sdfs_to_meshes_np(psrs, safe=False, rescale_verts=False):
    """
    - input:
        - psrs: b 1 r r r
    - return:
        - verts: list of (n 3)
        - faces: list of (m 3)
    """
    meshes = sdfs_to_meshes(psrs, safe=safe)
    vs1, fs1 = [m.vertices for m in meshes], [m.faces for m in meshes]
    vs2, fs2 = [], []
    for i in range(len(vs1)):
        v = (vs1[i] * 2 - 1) if rescale_verts else vs1[i]
        v = v.numpy(force=True) if isinstance(v, torch.Tensor) else v
        f = fs1[i].numpy(force=True) if isinstance(v, torch.Tensor) else fs1[i]
        vs2.append(v)
        fs2.append(f)
    return vs2, fs2


def sdf_to_point(sdf, n_points, safe=False):
    """
    - input:
        - sdf: 1 r r r
    - return:
        - point: n_points 3
    """
    if safe:
        try:
            mv, mf, mn = mc_from_psr(sdf, pytorchify=True)
            mesh = Meshes([mv], [mf], verts_normals=[mn])
            pts = sample_points_from_meshes(mesh, n_points)
        except RuntimeError:
            pts = sdf.new_zeros(1, n_points, 3)
    else:
        mv, mf, mn = mc_from_psr(sdf, pytorchify=True)
        mesh = Meshes([mv], [mf], verts_normals=[mn])
        pts = sample_points_from_meshes(mesh, n_points)

    return pts[0]


def sdfs_to_points(sdfs, n_points, safe=False):
    """
    - input:
        - sdfs: b 1 r r r
    - return:
        - points: b n_points 3
    """
    return torch.stack([sdf_to_point(sdf, n_points, safe=safe) for sdf in sdfs])


def sdf_to_point_fast(sdf, n_points):
    """
    - input:
        - sdf: 1 r r r
    - return:
        - point: n_points 3
    """
    mesh = cubify(-sdf, 0)
    pts = sample_points_from_meshes(mesh, n_points)
    return pts[0]


def sdfs_to_points_fast(sdfs, n_points):
    """
    - input:
        - sdfs: b 1 r r r
    - return:
        - points: b n_points 3
    """
    return torch.stack([sdf_to_point_fast(sdf, n_points) for sdf in sdfs])


def save_sdf_as_mesh(path, sdf, safe=False):
    """
    - input:
        - sdf: 1 r r r
    """
    verts, faces = sdfs_to_meshes_np(sdf[None], safe=safe)
    pcu.save_mesh_vf(str(path), verts[0], faces[0])


def plot_sdfs(sdfs, title=None, titles=None, safe=False, interactive=False, subplot_size=(5,5), fig_kwargs={}, view_kwargs={"azim": 30, "elev": 30, "roll": 0, "vertical_axis": "y"}):
    """
    - input:
        - sdfs: (b 1 r r r) or list of (1 r r r) or (b 1 r r r)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if interactive: 
        plt.ion()
    else:
        plt.ioff()

    # Ensure the sdfs are a tensor of shape (b, 1, r, r, r) 
    if isinstance(sdfs, torch.Tensor):
        num_cols = sdfs.shape[0]
        assert len(sdfs.shape) == 5 and sdfs.shape[1] == 1, f"sdfs must have shape (1, 1, r, r, r), got tensor with shape {sdfs.shape}."
    elif isinstance(sdfs, list): 
        num_cols = len(sdfs)
        assert all(len(x.shape) in {4, 5} for x in sdfs), f"All the sdfs in the list must have shape (1, r, r, r). Got shapes {[x.shape for x in sdfs]}."
        sdfs = torch.vstack(sdfs)
        if len(sdfs.shape) == 4:
            sdfs = sdfs.unsqueeze(1)

    # Create a figure with multiple subplots in one row
    num_sdfs = len(sdfs)
    if num_cols == num_sdfs:
        num_rows = 1
    else:
        if num_sdfs % num_cols == 0:
            num_rows = num_sdfs // num_cols
        else:
            import warnings
            warnings.warn(f"Number of total sdfs ({num_sdfs}) is not divisible by the number of element in the provided list ({num_cols}).")
            num_rows = num_sdfs
            num_cols = 1
            
    # Ensure titles is a list and has the same number of elements as the number of columns (if provided)
    if titles is not None:
        if isinstance(titles, str): 
            titles = [titles]
        assert len(titles) == 1 or len(titles) == num_cols, f"Number of titles must be 1 or equal to the number of elements of the list provided. Got {len(titles)} titles for {num_cols} columns."
        titles = np.array(titles)

    if subplot_size: 
        figsize = tuple(np.array(subplot_size) * np.array([num_cols, num_rows]))
        fig_kwargs.update({"figsize": figsize})
    fig, axes = plt.subplots(num_rows, num_cols, subplot_kw={'projection': '3d'}, squeeze=False, **fig_kwargs)

    meshes = sdfs_to_meshes(sdfs, safe=safe)

    for i in range(num_cols):
        for j in range(num_rows):
            idx = i * num_rows + j
            mesh = meshes[idx]
            verts, faces = mesh.vertices, mesh.faces
            axes[j, i].plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.6)
            axes[j, i].view_init(**view_kwargs)
            axes[j, i].set_xlim(0, 1)
            axes[j, i].set_ylim(0, 1)
            axes[j, i].set_zlim(0, 1)
            if titles is not None:
                if len(titles.flatten()) == 1:
                    axes[j, i].set_title(titles.flatten()[0])
                elif len(titles.flatten()) == num_cols:
                    axes[j, i].set_title(titles.flatten()[i])
                elif len(titles.flatten()) == num_sdfs:
                    axes[j, i].set_title(titles[i, j])
                else:
                    axes[j, i].set_title(f"SDF {idx}")

    if title:
        fig.suptitle(title)
    plt.show()



