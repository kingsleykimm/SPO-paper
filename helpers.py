"""Shared helpers for rl_continuous experiments."""

from acme import wrappers
import dm_env
import gym


_VALID_TASK_SUITES = ('gym', 'control')


def make_environment(suite: str, task: str) -> dm_env.Environment:
    """Makes the requested continuous control environment.

    Args:
        suite: One of 'gym' or 'control'.
        task: Task to load. If `suite` is 'control', the task must be formatted as
        f'{domain_name}:{task_name}'

    Returns:
        An environment satisfying the dm_env interface expected by Acme agents.
    """

    if suite not in _VALID_TASK_SUITES:
        raise ValueError(
            f'Unsupported suite: {suite}. Expected one of {_VALID_TASK_SUITES}')

    if suite == 'gym':
        env = gym.make(task, exclude_current_positions_from_observation=False)
        # Make sure the environment obeys the dm_env.Environment interface.
        env = wrappers.GymWrapper(env)

    elif suite == 'control':
        # environments used in paper are quadruped:walk, hopper:stand, walker:stand, walker:walk
        # Load dm_suite lazily not require Mujoco license when not using it.
        from dm_control import suite as dm_suite  # pylint: disable=g-import-not-at-top
        domain_name, task_name = task.split(':')
        env = dm_suite.load(domain_name, task_name)
        env = wrappers.ConcatObservationWrapper(env)

    # Wrap the environment so the expected continuous action spec is [-1, 1].
    # Note: this is a no-op on 'control' tasks.
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    return env