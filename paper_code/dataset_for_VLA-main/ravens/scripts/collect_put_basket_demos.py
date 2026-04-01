import os
import sys

from absl import app
from absl import flags
import cv2
import numpy as np

flags.DEFINE_string('assets_root', None, 'Ravens assets root')
flags.DEFINE_string('data_dir', '.', 'Output directory for dataset')
flags.DEFINE_string('obj_dir', '/media/ubuntu/sda/duan/OBJ3D', 'Directory with .glb/.obj mesh files')
flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_integer('n_per_object', 200, 'Successful episodes per object (total up to 6*n_per_object)')
flags.DEFINE_bool('disp', False, 'Show GUI')
flags.DEFINE_string('video_dir', None, 'If set, save one MP4 per successful episode (front camera view)')
flags.DEFINE_integer('video_fps', 5, 'FPS for exported MP4')

TARGET_OBJECTS = ('apple', 'pear', 'banana', 'elephant', 'deer', 'rhino')
FLAGS = flags.FLAGS

OBJECT_BASES = [
    ('apple', 'apple'), ('pear', 'pear'), ('banana', 'banana'),
    ('elephant', 'elephant'), ('deer', 'deer'), ('rhino', 'rhino'),
]
FIXED_SCALE = [1.0, 1.0, 1.0]
OBJECT_SCALES = {'apple': [0.001, 0.001, 0.001], 'banana': [0.0004, 0.0004, 0.0004],
                 'pear': [0.04, 0.04, 0.04], 'elephant': [0.06, 0.06, 0.06],
                 'deer': [0.03, 0.03, 0.03], 'rhino': [0.035, 0.035, 0.035]}
OBJECT_RPY_DEG = {
    'apple': (90.0, 0.0, 0.0), 'pear': (90.0, 0.0, 0.0), 'banana': (0.0, 0.0, 0.0),
    'elephant': (0.0, 180.0, 0.0), 'deer': (90.0, 0.0, 0.0), 'rhino': (90.0, 0.0, 0.0),
}


def _first_view(obs):
  c = obs.get('color')
  if c is None:
    return None
  if isinstance(c, (tuple, list)):
    return np.asarray(c[0])
  return np.asarray(c)


def _obs_to_bgr(obs):
  frame = _first_view(obs)
  if frame is None:
    return None
  if frame.ndim == 4:
    frame = frame[0]
  frame = np.uint8(frame)
  if frame.shape[2] == 3:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  return frame


def write_episode_mp4(episode, out_path, fps=5, step_intermediates=None):
  frames = []
  if step_intermediates is not None and len(step_intermediates) >= 1:
    f0 = _obs_to_bgr(episode[0][0])
    if f0 is not None:
      frames.append(f0)
    for i in range(len(step_intermediates)):
      for obs in step_intermediates[i]:
        f = _obs_to_bgr(obs)
        if f is not None:
          frames.append(f)
      f_next = _obs_to_bgr(episode[i + 1][0])
      if f_next is not None:
        frames.append(f_next)
  else:
    for obs, _, _, _ in episode:
      f = _obs_to_bgr(obs)
      if f is not None:
        frames.append(f)
  if not frames:
    return
  h, w = frames[0].shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
  for f in frames:
    writer.write(f)
  writer.release()


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
  script_dir = os.path.dirname(os.path.abspath(__file__))
  ravens_root = os.path.dirname(script_dir)
  assets_root = FLAGS.assets_root or os.path.join(ravens_root, 'ravens', 'environments', 'assets')
  assets_root = os.path.abspath(assets_root)
  obj_dir = os.path.abspath(FLAGS.obj_dir)
  if not os.path.isdir(obj_dir):
    print(f'obj_dir not found: {obj_dir}')
    sys.exit(1)
  objs = []
  for base, name in OBJECT_BASES:
    fname = find_mesh_in_dir(obj_dir, base)
    if fname:
      objs.append((fname, name, base))
    else:
      print(f'Missing: {base} in {obj_dir}')
  if len(objs) < 6:
    print(f'Need 6 objects, found {len(objs)}')
    sys.exit(1)
  object_config = [(obj_dir, f, OBJECT_SCALES.get(b, FIXED_SCALE), n, OBJECT_RPY_DEG.get(b, (0, 0, 0))) for f, n, b in objs]

  from ravens.tasks.put_object_in_basket import PutObjectInBasket
  from ravens.dataset import Dataset
  from ravens.environments.environment import Environment

  env = Environment(assets_root, disp=FLAGS.disp, hz=480)
  task = PutObjectInBasket(object_config=object_config)
  task.set_assets_root(assets_root)
  task.mode = FLAGS.mode
  agent = task.oracle(env)
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'put-object-in-basket-{task.mode}'))
  count = {obj: 0 for obj in TARGET_OBJECTS}

  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if task.mode == 'test' else -2
  max_steps = task.max_steps
  n_per = FLAGS.n_per_object

  def parse_instruction(instr):
    if not instr or not isinstance(instr, str):
      return None
    prefix = 'grasp '
    suffix = ' and put it into the basket'
    if instr.startswith(prefix) and instr.endswith(suffix):
      return instr[len(prefix):-len(suffix)].strip()
    return None

  while min(count.values()) < n_per:
    episode, total_reward = [], 0
    step_intermediates = [] if FLAGS.video_dir else None
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
    obs = env.reset()
    info = None
    reward = 0
    for _ in range(max_steps):
      act = agent.act(obs, info)
      episode.append((obs, act, reward, info))
      if FLAGS.video_dir:
        step_current = []
        env._step_frame_callback = lambda o, lst=step_current: lst.append(o)
      obs, reward, done, info = env.step(act)
      if FLAGS.video_dir:
        env._step_frame_callback = None
        step_intermediates.append(step_current)
      total_reward += reward
      if done:
        break
    episode.append((obs, None, reward, info))
    if total_reward > 0.99:
      instr = None
      for _, _, _, inf in episode:
        if inf and isinstance(inf, dict) and inf.get('instruction'):
          instr = inf['instruction']
          break
      obj = parse_instruction(instr)
      if obj in count and count[obj] < n_per:
        dataset.add(seed, episode)
        ep_id = dataset.n_episodes - 1
        if FLAGS.video_dir:
          os.makedirs(FLAGS.video_dir, exist_ok=True)
          mp4_path = os.path.join(FLAGS.video_dir, f'episode_{ep_id:06d}.mp4')
          write_episode_mp4(episode, mp4_path, fps=FLAGS.video_fps, step_intermediates=step_intermediates)
          print(f'  视频已导出: {mp4_path}')
        count[obj] += 1
        print(f'Episode {dataset.n_episodes} | {obj}: {count[obj]}/{n_per} | total: {sum(count.values())}')
    else:
      dist_xy = None
      if info and isinstance(info, dict):
        dist_xy = info.get('fail_dist_xy')
      if dist_xy is not None:
        print(f'  Failed (reward={total_reward:.3f}), object ~{dist_xy*100:.1f}cm from basket target, retrying...')
      else:
        print(f'  Failed (reward={total_reward:.3f}), retrying...')
  env.close()
  print(f'Done. Saved {dataset.n_episodes} episodes to {dataset.path}')
  for obj in TARGET_OBJECTS:
    print(f'  {obj}: {count[obj]}')


if __name__ == '__main__':
  app.run(main)
