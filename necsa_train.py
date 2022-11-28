# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os, datetime, json, argparse, copy
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

from abstracter import Abstracter, ScoreInspector

torch.backends.cudnn.benchmark = True


ALGO_NAME = 'NECSA_Drqv2'
# parser = argparse.ArgumentParser()

# parser.add_argument("--step", type=int, default=3)                  # Directory for storing all experimental data
# parser.add_argument("--grid_num", type=int, default=5)              # Directory for storing all experimental data
# parser.add_argument("--epsilon", type=float, default=0.1)            # Directory for storing all experimental data
# parser.add_argument("--raw_state_dim", type=int, default=50 ) 
# parser.add_argument("--state_dim", type=int, default=24) 
# parser.add_argument("--state_min", type=float, default=-1)        # 
# parser.add_argument("--state_max", type=float, default=1 )         # state_max, state_min
# parser.add_argument("--action_dim", type=int, default=6) 
# parser.add_argument("--action_min", type=float, default=-1 )        # 
# parser.add_argument("--action_max", type=float, default=1 )         # state_max, state_min
# parser.add_argument("--mode", type=str, default='hidden', choices=['state', 'state_action', 'hidden'] )   # 
# parser.add_argument("--reduction", action="store_true")   # 

# args=parser.parse_args()

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg, NECSA_DICT = None):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.inspector = ScoreInspector(
            3, 
            5, 
            50, 
            24, 
            -1, 
            1,
            6, 
            -1, 
            1, 
            'hidden', 
            True
            )

        self.abstracter = Abstracter(
                3, 
                0.1, 
            )

        self.abstracter.inspector = self.inspector

        self.time_step_list = []
        self.hidden_list = []
        self.reward_list = []
        self.done_list = []

        self.eval_results = []

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, _ = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        self.eval_results.append(total_reward / episode)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action, hidden = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)

            self.time_step_list.append(copy.deepcopy(time_step))
            self.hidden_list.append(hidden)
            self.reward_list.append(time_step.reward)
            self.done_list.append(time_step.last())
            print(hidden)
            print(time_step.reward)
            print(time_step.last())
            self.abstracter.append(hidden, time_step.reward, time_step.last())

            episode_reward += time_step.reward
            
            # self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

            if time_step.last():
                self.reward_list = self.abstracter.reward_shaping(np.array(self.hidden_list), np.array(self.reward_list))
                for i in range(len(self.reward_list)):
                    time_step = self.time_step_list[i]
                    print(time_step)
                    time_step = time_step._replace(reward=self.reward_list[i])
                    print(time_step)
                    exit()
                    self.replay_storage.add(time_step)

                self.time_step_list = []
                self.hidden_list = []
                self.reward_list = []

           

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='necsa_config')
def main(cfg):
    from necsa_train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()

    reward_save_path = 'results/' + workspace.cfg.task_name + '/' + ALGO_NAME.upper()
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    os.makedirs(reward_save_path, exist_ok=True)
    reward_save_path = reward_save_path + '/' + now + '.json'
    print(reward_save_path)
    with open(reward_save_path, 'w') as f:
        json.dump(workspace.eval_results, f)


if __name__ == '__main__':
    main()