import wandb
import numpy as np
import torch
import collections
import tqdm
import os
import copy
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points,seed = 73)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.output_dir = output_dir
        self.eval = True
        self.speedup_demo = False

    def run(self, policy: BasePolicy, save_video=False):
        seed = 73
        if self.eval == True:
            save_dir = os.path.join(self.output_dir,'evaldata')
            os.makedirs(save_dir, exist_ok=True)
            self.speedup_demo = False
        elif self.speedup_demo ==True:
            save_dir = os.path.join(self.output_dir,'data')
            os.makedirs(save_dir, exist_ok=True)
            load_data_path = os.path.join(self.output_dir,'evaldata')
            

        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env
        # print(self.n_action_steps)
        
        start = 0
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            load_data_path = os.path.join(self.output_dir,'evaldata')
            if self.speedup_demo ==True:
                load_data_path = os.path.join(load_data_path, f'delta_action_pos_{episode_idx}.npy')
                # 加载 .npy 文件中的数据
                action_abs_array = np.load(load_data_path)
                action_abs = action_abs_array[0:3,:]
            else:
                action_abs = None
            # print(action_abs.shape)
            # start rollout
          
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            idx = 0
            action_array = []
            obs_array = []
            while not done:
                # print(len(obs))
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    # print(obs_dict_input.shape)
                    # print(len(obs_dict_input['agent_pos']))
                    obs_array.extend(copy.deepcopy(obs_dict_input))
                    action_dict = policy.predict_action(obs_dict_input)
                    # print(len(action_dict)
                
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                # np_action_dict = np.array(np_action_dict)
                # print("np_action_dict",len(np_action_dict))
                # print(np_action_dict['action'].shape)
                action = np_action_dict['action'].squeeze(0)
                # action = self.Uniform_acceleration(action = action,speed = 2)
                idx+=len(action)
                # end+=len(action)
                # print("action",action.shape)
                action_array.extend(copy.deepcopy(action))
                # import pdb;pdb.set_trace()
                # print(f'action{action.shape},action_abs{action_abs.shape}')
                # print(f'action{action},action_abs{action_abs}')
                obs, reward, done, info = env.step(action,green_curve = action_abs)
                if self.speedup_demo ==True:
                    action_abs = action_abs_array[idx:3+idx,:]
                # print(info['target_pos'][0])
                # import pdb;pdb.set_trace()
                # print(len(info['target_pos'][0]))


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])
            print("is_success",is_success)
            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            
            print(len(info['target_pos'][0]))
            print(len(action_array))
            # import pdb;pdb.set_trace()
            if self.eval == True:   
                self.save_data(save_dir,episode_idx,info['target_pos'][0][start:],action_array,obs_array)
                start = len(info['target_pos'][0])
                print("start",start)
        
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data
    def Uniform_acceleration(self,action,speed = 2):
        # print(action.shape)
        action = action[::speed,:]# action大小是[201,4]取其中的[200/2,4]
        return action
    def save_data(self, save_dir, episode_idx, data_target_pos, data_action_array,data_obs_array):
        data_target_pos = np.array(data_target_pos)
        data_action_array = np.array(data_action_array) 
        data_act = np.array(data_action_array) 
        save_path_target_pos = os.path.join(save_dir, f'target_pos_{episode_idx}.npy')
        save_path_action_pos = os.path.join(save_dir, f'delta_action_pos_{episode_idx}.npy')
        save_path_obs_pos = os.path.join(save_dir, f'obs_pos_{episode_idx}.npy')
        save_path_act_pos = os.path.join(save_dir, f'delta_act_pos_{episode_idx}.npy')
        
        # 检查 data_target_pos d shape [200, 3] 并且 data_action_array 至少有 4 列，shape[201,4]
        if data_action_array.shape[0] == data_target_pos.shape[0] and data_action_array.shape[1] >= 4:
            data_action_array[:, :3] = data_target_pos  # 将data_action_array的前3列改成data_target_pos
        else:
            if data_action_array.shape[0] >= data_target_pos.shape[0]:
                data_action_array[:data_target_pos.shape[0], :3] = data_target_pos
                # 用 data_target_pos 的最后一行值补齐剩下的部分
                data_action_array[data_target_pos.shape[0]:, :3] = data_target_pos[-1, :]
            else:
                raise ValueError("data_target_pos has more rows than data_action_array.")
        
        np.save(save_path_target_pos, data_target_pos)
        print(data_action_array.shape)
        np.save(save_path_action_pos, data_action_array)
        np.save(save_path_obs_pos, data_obs_array)
        np.save(save_path_act_pos, data_act)

        print(f"Data saved to {len(data_action_array)}")
            # todo 将data save为 npy
        

