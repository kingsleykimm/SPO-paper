import acme
from typing import Optional
import optax
from acme.utils import counting
from acme.utils import loggers
import jnp
from acme.jax import networks as networks_lib
from acme import types
import jax

class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


class OfflineRLHF():
    # model_free approach
    pass
class RLHFLearner(acme.Learner):
    # nash RLHF uses state-action pairs, so we should do that with reward
    _state: TrainingState

    def __init__(
        self,
        networks: sac_networks.SACNetworks,
        rng: jnp.ndarray,
        iterator: Iterator[reverb.ReplaySample],
        policy_optimizer: optax.GradientTransformation,
        q_optimizer: optax.GradientTransformation,
        tau: float = 0.005,
        reward_scale: float = 1.0,
        discount: float = 0.99,
        entropy_coefficient: Optional[float] = None,
        target_entropy: float = 0,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None, 
        preference_model = None,
        mixture_coefficient : int = 0.5,
        num_sgd_steps_per_step: int = 1):

        self.preference_model = preference_model
        def actor_loss(current_policy_params: networks_lib.Params,
                       behavior_policy_params : networks_lib.Params,
                    alternative_policy_params : networks_lib.Params,
                   transitions: types.Transition,
                   key: networks_lib.PRNGKey) -> jnp.ndarray:
            cur_dist = networks.policy_network.apply(
                current_policy_params, transitions.observation) # softmax distribution after applying 
            cur_action = networks.sample(cur_dist, key)
            log_prob = networks.log_prob(cur_dist, cur_action)

            alternative_dist = networks.policy_network.apply(
                alternative_policy_params, transitions.observation
            )
            alternative_action = networks.sample(cur_dist, key)
            log_prob = networks.log_prob(alternative_dist, alternative_action)
            
            return jnp.mean(actor_loss)
        
        actor_grad = jax.value_and_grad(actor_loss)


# notes, x -. state, select actions based off that state