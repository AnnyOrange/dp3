import sys
sys.path.append('/home/xzj/project/origin-dp3/3D-Diffusion-Policy/third_party/gym-0.21.0')
import gym
import cv2
import numpy as np
from termcolor import cprint


class SimpleVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            steps_per_render=1,
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.steps_per_render = steps_per_render

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()

        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        self.step_count = 1
        return obs
    
    def step(self, action,green_act = None):
        entropy = action[-1]
        action = action[:-1]
        if green_act is None:
            result = super().step(action)
        else:
            # import pdb;pdb.set_trace()
            result = super().step(action,green_act)
        # print(action.shape)
        
        self.step_count += 1
        
        frame = self.env.render(mode=self.mode)
        frame = put_text(frame,  f"{entropy:.2e}")
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result
    
    def get_video(self):
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        return video
def put_text(img, text, is_waypoint=False, font_size=0.5, thickness=1, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if is_waypoint:
        img = cv2.putText(
            img,
            "*",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img