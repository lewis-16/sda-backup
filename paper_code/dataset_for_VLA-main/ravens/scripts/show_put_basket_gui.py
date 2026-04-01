"""
Show put-object-in-basket scene in PyBullet GUI (no data collection).
Supports OBJ (*_scaled.obj) or GLB (*.glb) in obj_dir. GLB is converted to OBJ+MTL with textures.
Usage: python scripts/show_put_basket_gui.py --obj_dir=/media/ubuntu/sda/duan/OBJ3D
"""
import os
import sys
import time

import numpy as np
from absl import app
from absl import flags

flags.DEFINE_string('assets_root', None, 'Ravens assets root (default: ravens/environments/assets)')
flags.DEFINE_string('obj_dir', '/media/ubuntu/sda/duan/OBJ3D', 'Directory containing mesh files (.obj or .glb)')
flags.DEFINE_string('single_object', None, 'Only load one object to check scale/orientation (e.g. apple)')
flags.DEFINE_bool('markers', False, 'Add small colored cubes at each object position to verify placement')
flags.DEFINE_bool('demo_layout', True, 'Use fixed 2x3 grid so all 6 objects stay in view')
flags.DEFINE_bool('aabb_boxes', False, 'Draw semi-transparent AABB box per object (see all 6 if mesh not rendering)')
flags.DEFINE_integer('hold_sec', 60, 'Seconds to keep GUI open (0 = until key interrupt)')
FLAGS = flags.FLAGS

OBJECT_BASES = [
    ('apple', 'apple'),
    ('pear', 'pear'),
    ('banana', 'banana'),
    ('elephant', 'elephant'),
    ('deer', 'deer'),
    ('rhino', 'rhino'),
]
FIXED_SCALE = [1.0, 1.0, 1.0]
SCALE_1_100 = [0.01, 0.01, 0.01]
OBJECT_SCALES = {'apple': [0.001, 0.001, 0.001], 'banana': [0.0004, 0.0004, 0.0004],
                 'pear': [0.04, 0.04, 0.04], 'elephant': [0.06, 0.06, 0.06],
                 'deer': [0.3, 0.3, 0.3], 'rhino': [0.035, 0.035, 0.035]}
OBJECT_RPY_DEG = {
    'apple': (90.0, 0.0, 0.0),
    'pear': (90.0, 0.0, 0.0),
    'banana': (00.0, 00.0, 0.0),
    'elephant': (0.0, 180.0, 0.0),
    'deer': (90.0, 0.0, 0.0),
    'rhino': (90.0, 0.0, 0.0),
}


def find_mesh_in_dir(obj_dir, base_name):
  for ext in ('.glb', '_scaled.obj', '.obj'):
    path = os.path.join(obj_dir, base_name + ext)
    if os.path.isfile(path):
      return base_name + ext
  sub = os.path.join(obj_dir, base_name)
  if os.path.isdir(sub):
    for ext in ('.glb', '_scaled.obj', '.obj'):
      path = os.path.join(sub, base_name + ext)
      if os.path.isfile(path):
        return os.path.join(base_name, base_name + ext)
  return None


def main(_):
  assets_root = FLAGS.assets_root
  if assets_root is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ravens_root = os.path.dirname(script_dir)
    assets_root = os.path.join(ravens_root, 'ravens', 'environments', 'assets')
  assets_root = os.path.abspath(assets_root)
  obj_dir = os.path.abspath(FLAGS.obj_dir)

  if not os.path.isdir(obj_dir):
    print(f'OBJ dir not found: {obj_dir}')
    sys.exit(1)
  objs = []
  if FLAGS.single_object:
    base = FLAGS.single_object.strip().lower()
    name_map = dict((b, n) for b, n in OBJECT_BASES)
    name = name_map.get(base, base)
    fname = find_mesh_in_dir(obj_dir, base)
    if fname:
      objs = [(fname, name, base)]
    else:
      print(f'Not found in {obj_dir}: {base}.glb / {base}_scaled.obj / {base}.obj')
      sys.exit(1)
    print(f'Single object mode: {name} ({fname})', flush=True)
  else:
    for base, name in OBJECT_BASES:
      fname = find_mesh_in_dir(obj_dir, base)
      if fname:
        objs.append((fname, name, base))
      else:
        print(f'Not found in {obj_dir}: {base}.glb / {base}_scaled.obj / {base}.obj')
    if len(objs) < 6:
      print(f'Found {len(objs)}/6 objects. Need one of (.glb, _scaled.obj, .obj) per base: {[x[0] for x in OBJECT_BASES]}')
      sys.exit(1)

  object_config = [(obj_dir, fname, OBJECT_SCALES.get(base, FIXED_SCALE), name, OBJECT_RPY_DEG.get(base, (0.0, 0.0, 0.0))) for fname, name, base in objs]

  demo_poses = None
  if FLAGS.demo_layout and len(objs) >= 1:
    from ravens.utils import utils
    z_link = 1.0
    if len(objs) == 1:
      _, _, base = objs[0]
      rpy_deg = OBJECT_RPY_DEG.get(base, (0.0, 0.0, 0.0))
      rpy_rad = np.deg2rad(rpy_deg)
      quat = utils.eulerXYZ_to_quatXYZW(rpy_rad)
      demo_poses = [((0.5, 0.0, z_link), quat)]
    else:
      xs = [0.35, 0.45, 0.55]
      ys = [-0.15, 0.15]
      demo_poses = []
      idx = 0
      for y in ys:
        for x in xs:
          _, _, base = objs[idx] if idx < len(objs) else (None, None, None)
          rpy_deg = OBJECT_RPY_DEG.get(base, (0.0, 0.0, 0.0))
          rpy_rad = np.deg2rad(rpy_deg)
          quat = utils.eulerXYZ_to_quatXYZW(rpy_rad)
          demo_poses.append(((x, y, z_link), quat))
          idx += 1

  from ravens.tasks.put_object_in_basket import PutObjectInBasket
  from ravens.environments.environment import Environment

  env = Environment(assets_root, disp=True, hz=240)
  import pybullet as p
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
  p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
  task = PutObjectInBasket(object_config=object_config, demo_poses=demo_poses)
  task.set_assets_root(assets_root)
  env.set_task(task)
  print('Resetting scene (loading OBJs may take a while)...', flush=True)
  obs = env.reset()
  n_rigid = len(env.obj_ids['rigid'])
  print(f'Scene loaded: {n_rigid} object(s) on table (plus 1 basket).', flush=True)
  if n_rigid > 0:
    p.resetDebugVisualizerCamera(
        cameraDistance=0.9,
        cameraYaw=25,
        cameraPitch=-32,
        cameraTargetPosition=[0.5, 0, 1.0])
  print('Camera: Right-drag=rotate, Middle-drag=pan, Scroll=zoom. Close window or wait.', flush=True)
  if n_rigid == 0:
    print('No objects appeared - check OBJ paths and *_scaled.obj names.', flush=True)
  elif n_rigid < 6:
    print(f'Only {n_rigid}/6 objects loaded. Check terminal for "Skip ..." messages above.', flush=True)
  if n_rigid > 0:
    for i, idx in enumerate(env.obj_ids['rigid']):
      pos, _ = p.getBasePositionAndOrientation(idx)
      aabb_min, aabb_max = p.getAABB(idx)
      sx = aabb_max[0] - aabb_min[0]
      sy = aabb_max[1] - aabb_min[1]
      sz = aabb_max[2] - aabb_min[2]
      name = objs[i][1] if i < len(objs) else str(i)
      print(f'  [{name}] x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.3f}  size=({sx:.3f},{sy:.3f},{sz:.3f})', flush=True)
      if sx < 0.001 or sy < 0.001 or sz < 0.001:
        print(f'    -> mesh likely failed to load (near-zero AABB)', flush=True)
      if max(sx, sy, sz) > 0.5:
        print(f'    -> mesh very large (>{0.5}m), may block view; consider re-scaling OBJ to ~10cm', flush=True)
    if FLAGS.markers and n_rigid > 0:
      col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015])
      colors = [(1, 0, 0), (1, 1, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 0.5, 0)]
      for i, idx in enumerate(env.obj_ids['rigid']):
        pos, _ = p.getBasePositionAndOrientation(idx)
        vis_i = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], rgbaColor=[*colors[i % 6], 1])
        bid = p.createMultiBody(0, col, vis_i, [pos[0], pos[1], pos[2] + 0.10])
      print(f'  {n_rigid} colored cube(s) added 10cm above each object.', flush=True)
    if FLAGS.aabb_boxes and n_rigid > 0:
      half = 0.05
      colors = [(1, 0, 0), (1, 1, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 0.5, 0)]
      col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
      for i, idx in enumerate(env.obj_ids['rigid']):
        pos, _ = p.getBasePositionAndOrientation(idx)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[*colors[i % 6], 0.6])
        p.createMultiBody(0, col, vis, [pos[0], pos[1], pos[2] + half])
      print(f'  {n_rigid} fixed-size box(es) at object base(s). Disable with --noaabb_boxes if mesh visible.', flush=True)
  if FLAGS.hold_sec > 0:
    time.sleep(FLAGS.hold_sec)
  else:
    try:
      while True:
        time.sleep(1)
    except KeyboardInterrupt:
      pass
  env.close()


if __name__ == '__main__':
  app.run(main)
