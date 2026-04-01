# coding=utf-8
# Copyright 2024 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Put specified object in basket. Language-conditioned pick-place."""

import os
import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils


DEFAULT_OBJECT_CONFIG = [
    ('kitting', '00.obj', [0.003, 0.003, 0.001], 'apple'),
    ('kitting', '01.obj', [0.003, 0.003, 0.001], 'pear'),
    ('kitting', '02.obj', [0.003, 0.003, 0.001], 'banana'),
    ('kitting', '03.obj', [0.003, 0.003, 0.001], 'elephant'),
    ('kitting', '04.obj', [0.003, 0.003, 0.001], 'deer'),
    ('kitting', '05.obj', [0.003, 0.003, 0.001], 'rhino'),
]

BASKET_URDF = 'bowl/bowl.urdf'
BASKET_SIZE = (0.12, 0.12, 0)
BASKET_PLACE_OFFSET = (0, 0, 0.04)
TABLE_Z = 1.0
BASKET_FIXED_POSE = ((0.5, 0.4, TABLE_Z), (0.0, 0.0, 0.0, 1.0))
BASKET_XY = (0.5, 0.4)
MIN_DIST_CM = 0.1


class PutObjectInBasket(Task):
  """Pick one of 6 objects and place it in the basket by instruction."""

  def __init__(self, *args, **kwargs):
    self.object_config = kwargs.pop('object_config', DEFAULT_OBJECT_CONFIG)
    self.mesh_origin_rpy = kwargs.pop('mesh_origin_rpy', None)
    self.demo_poses = kwargs.pop('demo_poses', None)
    super().__init__(*args, **kwargs)
    self.max_steps = 1
    self.pos_eps = 0.05
    self.instruction = None

  def reset(self, env):
    super().reset(env)
    import pybullet as p
    table_pose = self.get_random_pose(env, (0.01, 0.01, 0.001))
    if table_pose[0] is not None:
      table_z = float(table_pose[0][2])
    else:
      table_z = TABLE_Z
    basket_pose = ((0.5, 0.4, table_z), (0.0, 0.0, 0.0, 1.0))
    env.add_object(BASKET_URDF, basket_pose, 'fixed')
    if env.obj_ids['fixed']:
      basket_aabb_min, _ = p.getAABB(env.obj_ids['fixed'][0])
      table_z = float(basket_aabb_min[2])
    place_pos = np.asarray(utils.apply(basket_pose, BASKET_PLACE_OFFSET))
    place_quat = np.asarray((0, 0, 0, 1))
    place_pose = (place_pos, place_quat)

    template = 'kitting/object-template.urdf'
    colors = [
        utils.COLORS['red'], utils.COLORS['yellow'], utils.COLORS['blue'],
        utils.COLORS['green'], utils.COLORS['purple'], utils.COLORS['orange']
    ]
    import tempfile
    import string
    obj_ids = []
    def dist_xy(a, b):
      return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    for i, entry in enumerate(self.object_config):
      folder = entry[0]
      fname = entry[1]
      scale = entry[2]
      name = entry[3]
      rpy_deg = entry[4] if len(entry) > 4 else None
      size = (0.08, 0.08, 0.02)
      if self.demo_poses is not None and i < len(self.demo_poses):
        pose = self.demo_poses[i]
      else:
        pose = None
        for _ in range(30):
          candidate = self.get_random_pose(env, size)
          if candidate[0] is None or candidate[1] is None:
            continue
          pos_xy = (candidate[0][0], candidate[0][1])
          if dist_xy(pos_xy, BASKET_XY) < MIN_DIST_CM:
            continue
          ok = True
          for j in range(len(env.obj_ids['rigid'])):
            other_pos, _ = p.getBasePositionAndOrientation(env.obj_ids['rigid'][j])
            if dist_xy(pos_xy, (other_pos[0], other_pos[1])) < MIN_DIST_CM:
              ok = False
              break
          if ok:
            if rpy_deg is not None:
              quat = utils.eulerXYZ_to_quatXYZW(np.deg2rad(rpy_deg))
              pose = (candidate[0], quat)
            else:
              pose = candidate
            break
        if pose is None or pose[0] is None:
          print(f'Skip {fname}: no valid pose (min dist {MIN_DIST_CM}m)', flush=True)
          continue
        if not fname.lower().endswith('.urdf') and os.path.isabs(folder):
          pos, rot = pose
          pos = (pos[0], pos[1], pos[2] + 0.12)
          pose = (pos, rot)
      try:
        if fname.lower().endswith('.urdf'):
          urdf_path = os.path.join(folder, fname)
          obj_id = env.add_object(urdf_path, pose)
        else:
          if os.path.isabs(folder):
            import shutil
            tmpdir = tempfile.mkdtemp(prefix='ravens_obj_')
            src = os.path.join(folder, fname)
            dst = os.path.join(tmpdir, fname)
            if not os.path.isfile(src):
              print(f'Skip {fname}: mesh not found at {src}', flush=True)
              continue
            try:
              os.makedirs(os.path.dirname(dst), exist_ok=True)
              shutil.copy2(src, dst)
            except Exception as e:
              print(f'Skip {fname}: copy failed {e}', flush=True)
              continue
            mesh_fname = fname
            if fname.lower().endswith('.glb'):
              try:
                import trimesh
                scene = trimesh.load(dst)
                mesh_fname = os.path.splitext(fname)[0] + '.obj'
                out_obj = os.path.join(tmpdir, mesh_fname)
                os.makedirs(os.path.dirname(out_obj), exist_ok=True)
                scene.export(out_obj)
              except Exception as e:
                print(f'Skip {fname}: GLB->OBJ failed {e}', flush=True)
                continue
            mesh_path_for_urdf = mesh_fname
            urdf_out_dir = tmpdir
          else:
            mesh_path_for_urdf = os.path.join(self.assets_root, folder, fname)
            urdf_out_dir = None
          color = (1.0, 1.0, 1.0) if fname.lower().endswith('.glb') else colors[i % len(colors)]
          replace = {'FNAME': (mesh_path_for_urdf,), 'SCALE': scale,
                     'COLOR': color}
          full_template = os.path.join(self.assets_root, template)
          with open(full_template, 'r') as f:
            fdata = f.read()
          for field in replace:
            for j in range(len(replace[field])):
              fdata = fdata.replace(f'{field}{j}', str(replace[field][j]))
          if self.mesh_origin_rpy is not None and urdf_out_dir:
            rpy_str = ' '.join(str(x) for x in self.mesh_origin_rpy)
            fdata = fdata.replace('rpy="0 0 0"', 'rpy="{}"'.format(rpy_str))
          rname = ''.join(__import__('random').choices(string.ascii_lowercase + string.digits, k=8))
          if urdf_out_dir:
            urdf = os.path.join(urdf_out_dir, f'_ravens_{rname}.urdf')
          else:
            urdf = os.path.join(tempfile.gettempdir(), f'object-template.{rname}.urdf')
          with open(urdf, 'w') as f:
            f.write(fdata)
          obj_id = env.add_object(urdf, pose)
          if os.path.isabs(folder):
            aabb_min, _ = p.getAABB(obj_id)
            bottom_z = aabb_min[2]
            dz = table_z - bottom_z
            if abs(dz) > 1e-4:
              cur_pos, cur_rot = p.getBasePositionAndOrientation(obj_id)
              new_pos = (cur_pos[0], cur_pos[1], cur_pos[2] + dz)
              p.resetBasePositionAndOrientation(obj_id, new_pos, cur_rot)
          if urdf_out_dir is None:
            try:
              os.remove(urdf)
            except OSError:
              pass
        obj_ids.append((obj_id, name))
      except Exception as e:
        print(f'Skip {fname}: {e}')
        continue

    if len(obj_ids) < 1:
      self.goals = []
      return

    target_idx = np.random.randint(0, len(obj_ids))
    target_id, target_name = obj_ids[target_idx]
    self.instruction = f'grasp {target_name} and put it into the basket'

    self.goals.append(([(target_id, (0, None))], np.int32([[1]]),
                       [place_pose], False, False, 'pose', None, 1))

  def reward(self):
    import pybullet as p
    reward, info = super().reward()
    if hasattr(self, 'instruction') and self.instruction is not None:
      info['instruction'] = self.instruction
    if reward <= 0 and self.goals:
      objs, _, targs = self.goals[0][:3]
      if objs and targs:
        obj_id = objs[0][0]
        try:
          pos, _ = p.getBasePositionAndOrientation(obj_id)
          tpos = np.asarray(targs[0][0])
          dist_xy = np.linalg.norm(np.float32(pos[:2]) - np.float32(tpos[:2]))
          info['fail_dist_xy'] = float(dist_xy)
        except Exception:
          pass
    return reward, info
