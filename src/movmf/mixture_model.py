"""Finite mixture model classes"""

from abc import ABC, abstractmethod
from copy import deepcopy
from tqdm.auto import trange

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
from jax import jit, lax, value_and_grad, vmap
from jax.tree_util import register_pytree_node_class, tree_map, tree_leaves
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

@register_pytree_node_class
class Parameter:
    """A lightweight wrapper for parameters of a model. It combines the `value`
    (a JAX PyTree) with a flag `is_frozen` (bool) to specify whether or not
    the parameter should be updated during model learning, as well as a `bijector`
    (tensorflow_probability.bijectors.Bijector) to map the parameter to/from an
    unconstrained space. Borrowed from earlier versions of probml/ssm_jax code
    """

    def __init__(self, value, is_frozen=False, bijector=None):
        self.value = value
        self.is_frozen = is_frozen
        self.bijector = bijector if bijector is not None else tfb.Identity()

    def __repr__(self):
        return f"Parameter(value={self.value}, " \
               f"is_frozen={self.is_frozen}, " \
               f"bijector={self.bijector}"

    @property
    def unconstrained_value(self):
        return self.bijector(self.value)

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    def tree_flatten(self):
        children = (self.value,)
        aux_data = self.is_frozen, self.bijector
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)

class FiniteMixtureModel(ABC):
    r"""
    A base class for exponential family finite mixture models. This base
    class standardizes parameters and methods for fitting model to data.

    The generative formulation of the finite mixture model is

    .. math::
        z_m ~ \textrm{Cat}(\alpha_m) for m=1,\ldots,M
        x_i | z_m ~ p(x | z_m, \theta_m)  for i =1,\ldots,N

    for mixture membership (mixing) weights :math:`\alpha_m`, and mixture
    component parameters :math:`\theta_m`.
    """

    def __init__(self, mixing_probs):
        """
        Abstract base class specifies mixing probabilities. Child class
        specifies the component distribution.

        Params
            mixing_probs[...,m]: probabilities of Categorical distributions.
        """
        self._mixing_probs = Parameter(mixing_probs, bijector=tfb.Invert(tfb.SoftmaxCentered()))
    
    # ----------------------------------------------
    # Definition model Parameters and distributions
    @property
    def params(self,):
        """Return list of all (unfrozen) parameters."""
        items = sorted(self.__dict__.items())
        param_values = [param.value for _, param in items if (isinstance(param, Parameter) and not param.is_frozen)]
        return param_values
    
    @params.setter
    def params(self, values):
        items = sorted(self.__dict__.items())
        params = [param for _, param in items if isinstance(param, Parameter) and not param.is_frozen]
        assert len(params) == len(values)
        for param, value in zip(params, values):
            param.value = value

    def mixing_distribution(self,) ->tfd.Distribution:
        """Return the distribution of the mixture."""
        alphas = self._mixing_probs.value
        return tfd.Categorical(probs=alphas, name='mixing_distr')

    @abstractmethod
    def component_distribution(self, m) -> tfd.Distribution:
        """Return the distribution of the m-th mixture component."""
        raise NotImplementedError

    # ------------------------------------------------------------------------
    # Generic implementation of tree_flatten and unflatten. This assumes that
    # the Parameters are all valid JAX PyTree nodes.
    def tree_flatten(self):
        items = sorted(self.__dict__.items())
        param_values = [val for key, val in items if isinstance(val, Parameter)]
        param_names = [key for key, val in items if isinstance(val, Parameter)]
        return param_values, param_names

    @classmethod
    def tree_unflatten(cls, aux_data, param_values):
        param_names = aux_data
        obj = object.__new__(cls)
        for name, value in zip(param_names, param_values):
            setattr(obj, name, value)
        return obj

    # -----------------------
    # Initialization methods
    @classmethod
    @abstractmethod
    def initialize_random(cls, seed, n_mixtures, component_dim):
        """Initialize mixture model randomly."""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def initialize_kmeans(cls, seed, n_mixtures, observations, n_iters=10, tol=1e-3):
        """Initialize mixture model from data with k-means algorithm."""
        raise NotImplementedError

    # ---------------------
    # Distribution methods
    def log_prob(self, observations):
        r"""Calculate log likelihood of observations under the mixture model.

        Recall that
        .. math ::
            \log p(x_i) = \log (\sum_m p(z_i=m) * \exp\{\log p(x_i | z_i=m))\}))
                        = \log (\sum_m \exp\{\log p(x_i | z_i=m) + \log p(z_i=m))\}))
        """

        alphas = self._mixing_probs.value
        M = alphas.shape[-1]
        
        lps = vmap(lambda m:
            self.component_distribution(m).log_prob(observations), out_axes=-1,
        )(jnp.arange(M))
        lps += jnp.log(alphas)

        return jnp.sum(logsumexp(lps, axis=-1))

    def sample(self, seed, sample_shape=()):
        """Draw samples and assignments from the model
        
        Params
            seed (jr.PRNGKey)
            sample_shape (tuple)

        Returns
            assgns[...,]
            samples[...,d]
        """
        seed_mix, seed_comp, seed_shuffle = jr.split(seed, 3)

        M = self._mixing_probs.value.shape[-1]

        # Draw samples from mixing distr and count number of draws for each mixture
        # i.e. draw number of samples from a multinomial distribution
        assgns = self.mixing_distribution().sample(sample_shape, seed_mix)
        counts = vmap(lambda m: jnp.sum(assgns==m))(jnp.arange(M))

        # Draw specified number of samples from each mixture sequentially
        assignments = jnp.concatenate([
            jnp.ones(n_samples, dtype=int) * m for m, n_samples in enumerate(counts)
        ])
        samples = jnp.concatenate([
            self.component_distribution(m).sample((n_samples,), jr.fold_in(seed_comp, m))
            for m, n_samples in enumerate(counts)
        ])

        # Shuffle samples so that they are no longer grouped by mixture
        shuffled_indices = jr.permutation(seed_shuffle, len(samples))
        shuffled_samples = samples[shuffled_indices]
        shuffled_assignments = assignments[shuffled_indices]
        return (shuffled_assignments.reshape(*sample_shape), 
               shuffled_samples.reshape(*sample_shape, -1))

    # --------------
    # EM algorithm
    @abstractmethod
    def _m_step_component(self, observations, expected_assignments):
        """Weighted posterior parameters of mixture distribution."""
        raise NotImplementedError

    def m_step(self, observations, expected_assignments, ):
        """Calculate maximum likelihood estimate of mixture weights and
        component distribution parameters. Update parameters in-place.

        Parameters
            expected_assignments[n,m]
            observations[n,d]:
        """
        self._mixing_probs.value = jnp.mean(expected_assignments, axis=0)
        self._m_step_component(observations, expected_assignments)
        return

    def e_step(self, observations):
        """Calculate expected membership weights for each observation.

        Params
            observations[...,d]

        Returns 
            expected_assignments[...,m]: Each row is M-length vector summing to 1.
        """

        alphas = self._mixing_probs.value
        M = len(alphas)

        lps = vmap(lambda m:
            self.component_distribution(m).log_prob(observations), out_axes=-1,
        )(jnp.arange(M))
        lps += jnp.log(alphas)
        
        log_sum = logsumexp(lps, axis=-1, keepdims=True)
        return jnp.exp(lps - log_sum)

    def fit_em(self, observations, n_iters=100):
        """Fits specified weights and distribution parameters to observations
        
        Parameters
            observations[...,d]: Samples assumed to be drawn iid from p(x)
            n_iters: int. Number of EM iterations to run.

        Return
            log_probs[n]: log probability 
        """
        
        def em_step(carry, i):
            self.params = carry
            expected_assignments = self.e_step(observations)
            self.m_step(observations, expected_assignments)
            lp = self.log_prob(observations)
            return self.params, lp
 
        params, log_probs = lax.scan(em_step, self.params, jnp.arange(n_iters))
        log_probs.block_until_ready()
        self.params = params

        return log_probs

# =============================================================================
# From https://www.tensorflow.org/probability/examples/
# TensorFlow_Probability_Case_Study_Covariance_Estimation
PSDToRealBijector = tfb.Chain(
    [
        tfb.Invert(tfb.FillTriangular()),
        tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
        tfb.Invert(tfb.CholeskyOuterProduct()),
    ]
)

@register_pytree_node_class
class GaussianMixtureModel(FiniteMixtureModel):
    """Finite mixture model with Gaussian component distributions."""

    def __init__(self, mixing_probs, means, covariances):
        super().__init__(mixing_probs)

        self._component_means = Parameter(means)
        self._component_covariances = Parameter(covariances, bijector=PSDToRealBijector)

    @classmethod
    def initialize_random(cls, seed, n_mixtures, component_dim):
        """Initialize mixture model randomly.
        
        Params
            seed (jr.PRNGKey):
            n_mixtures (int):
            component_dim (int): event shape of component distribution
        """

        seed_cat, seed_mean, seed_cov = jr.split(seed, 3)
        mixing_probs = jr.dirichlet(seed_cat, jnp.ones(n_mixtures))
        component_means = jr.normal(seed_mean, (n_mixtures, component_dim))
        component_covs = jnp.tile(jnp.eye(component_dim), (n_mixtures, 1, 1))
        
        return cls(mixing_probs, component_means, component_covs)
    
    @classmethod
    def initialize_kmeans(cls, seed, n_mixtures, observations, n_iters=10, tol=1e-3):
        # TODO Update function string in parent class as need
        """Initialize mixture model from data with k-means algorithm."""
        raise NotImplementedError

    def component_distribution(self, mixture) -> tfd.Distribution:
        mean = self._component_means.value[mixture]
        cov = self._component_covariances.value[mixture]
        return tfd.MultivariateNormalFullCovariance(mean, cov)
    
    def _m_step_component(self, observations, expected_assignments):
        """Update model with weighted posterior parameters.
        
        Params
            observations[....,d]
            expected_assignments[...,m]
        """
        D = observations.shape[-1]

        # Calculate normalized membership weights
        # Catch DBZ errors that occur when there are no assignments to a mixture
        total_weights = jnp.sum(expected_assignments, axis=0)
        normd_weights = jnp.where(
            total_weights[None,:,] > 1e-6,
            expected_assignments/total_weights,
            0.
        )

        # Calculate (normalized) weighted sufficient statistics
        normd_x = jnp.einsum('...m, ...d -> md', normd_weights, observations)
        normd_xxT = jnp.einsum('...m, ...d, ...e -> mde', normd_weights, observations, observations)
        normd_scatter = jnp.einsum('...md, ...me -> mde', normd_x, normd_x)

        # Update parameters
        self._component_means.value = normd_x
        self._component_covariances.value = normd_xxT - normd_scatter + jnp.eye(D) * 1e-4
        
        return
