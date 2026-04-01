import os
import pickle
import sys

from absl import app
from absl import flags
import numpy as np

flags.DEFINE_string('data_dir', None, 'Ravens dataset dir, e.g. data/put-object-in-basket-train')
flags.DEFINE_string('out_dir', None, 'Output dir for OpenVLA-style export')
flags.DEFINE_integer('max_episodes', None, 'Max episodes to export (default all)')
FLAGS = flags.FLAGS


def main(_):
  data_dir = os.path.abspath(FLAGS.data_dir)
  out_dir = os.path.abspath(FLAGS.out_dir or (data_dir + '-openvla'))
  if not os.path.isdir(data_dir):
    print(f'data_dir not found: {data_dir}')
    sys.exit(1)
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from ravens.dataset import Dataset

  dataset = Dataset(data_dir)
  n = dataset.n_episodes
  if FLAGS.max_episodes is not None:
    n = min(n, FLAGS.max_episodes)
  os.makedirs(out_dir, exist_ok=True)

  for ep_id in range(n):
    episode, _ = dataset.load(ep_id, images=True)
    instruction = None
    for _, _, _, info in episode:
      if info and isinstance(info, dict) and info.get('instruction'):
        instruction = info['instruction']
        break
    if instruction is None:
      instruction = 'grasp object and put it into the basket'
    ep_dir = os.path.join(out_dir, f'episode_{ep_id:06d}')
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, 'instruction.txt'), 'w') as f:
      f.write(instruction)
    actions = []
    for t, (obs, act, _, _) in enumerate(episode):
      if obs.get('color') is not None:
        img = obs['color']
        if isinstance(img, (tuple, list)):
          img = np.asarray(img[0])
        elif len(img.shape) == 4:
          img = img[0]
        np.save(os.path.join(ep_dir, f'image_{t:04d}.npy'), np.asarray(img))
      if act is not None:
        pose0 = act.get('pose0')
        pose1 = act.get('pose1')
        if pose0 is not None and pose1 is not None:
          actions.append({
            'pose0': (np.asarray(pose0[0]).tolist(), np.asarray(pose0[1]).tolist()),
            'pose1': (np.asarray(pose1[0]).tolist(), np.asarray(pose1[1]).tolist()),
          })
    with open(os.path.join(ep_dir, 'actions.pkl'), 'wb') as f:
      pickle.dump(actions, f)
  print(f'Exported {n} episodes to {out_dir}')


if __name__ == '__main__':
  app.run(main)
