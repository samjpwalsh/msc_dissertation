import scipy
import tensorflow as tf
from keras import layers
import os
from typing import Callable, Optional
from gymnasium import logger
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def mlp(x, sizes, activation, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)

def logprobabilities(logits, action, num_actions):
    # Compute the log-probabilities of taking actions by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(action, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability

def confidence_interval(data, conf=0.95):
    n = len(data)
    se = scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + conf) / 2, n-1)
    return h


# Code below is from https://gymnasium.farama.org/main/_modules/gymnasium/utils/save_video/ with small modifications

def capped_cubic_video_schedule(episode_id: int) -> bool:
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

def save_video(
    frames: list,
    video_folder: str,
    episode_trigger: Callable[[int], bool] = None,
    step_trigger: Callable[[int], bool] = None,
    video_length: Optional[int] = None,
    name_prefix: str = "rl-video",
    episode_index: int = 0,
    step_starting_index: int = 0,
    **kwargs,
):
    if not isinstance(frames, list):
        logger.error(f"Expected a list of frames, got a {type(frames)} instead.")
    if episode_trigger is None and step_trigger is None:
        episode_trigger = capped_cubic_video_schedule

    video_folder = os.path.abspath(video_folder)
    os.makedirs(video_folder, exist_ok=True)
    path_prefix = f"{video_folder}/{name_prefix}"

    if episode_trigger is not None and episode_trigger(episode_index):
        clip = ImageSequenceClip(frames[:video_length], **kwargs)
        clip.write_videofile(f"{path_prefix}-episode-{episode_index}.mp4", verbose=False, logger=None)

    if step_trigger is not None:
        # skip the first frame since it comes from reset
        for step_index, frame_index in enumerate(
            range(1, len(frames)), start=step_starting_index
        ):
            if step_trigger(step_index):
                end_index = (
                    frame_index + video_length if video_length is not None else None
                )
                clip = ImageSequenceClip(frames[frame_index:end_index], **kwargs)
                clip.write_videofile(f"{path_prefix}-step-{step_index}.mp4", verbose=False, logger=None)
