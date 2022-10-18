import pytest

import jax.numpy as jnp
import jax.random as jr

from movmf import GaussianMixtureModel

def generate_observations(seed, n_mixtures, n_dims, n_samples=100):
    seed_cat, seed_mean, seed_cov, seed_sample = jr.split(seed, 4)
    mixing_probs = jr.dirichlet(seed_cat, jnp.ones(n_mixtures))
    component_means = jr.normal(seed_mean, (n_mixtures, n_dims))

    scales = jr.uniform(seed_cov, (n_mixtures,), minval=1.e-3, maxval=3.)
    component_covs = jnp.tile(jnp.eye(n_dims), (n_mixtures, 1, 1))
    component_covs *= scales[:, None, None]
    
    true_gmm = GaussianMixtureModel(mixing_probs, component_means, component_covs)
    true_assgns, samples = true_gmm.sample(seed_sample, (n_samples,))
    return true_gmm, true_assgns, samples

def test_log_prob(seed=jr.PRNGKey(1510), n_mixtures=5, n_dims=3, n_samples=100):
    true_gmm, _, observations = generate_observations(seed, n_mixtures, n_dims, n_samples)

    assert jnp.isclose(-465.6154, true_gmm.log_prob(observations), atol=1e-3)

def test_fit_em(seed=jr.PRNGKey(1609), n_mixtures=5, n_dims=2, n_samples=500):
    seed_true, seed_test = jr.split(seed)
    true_gmm, _, observations = generate_observations(seed_true, n_mixtures, n_dims, n_samples)

    test_gmm = GaussianMixtureModel.initialize_random(seed_test, n_mixtures, n_dims)
    lps = test_gmm.fit_em(observations, n_iters=10)

    # TODO Implement permutation function to find permutation which minimizes
    # test_gmm means to true_gmm means.
    assert lps[-1] > lps[0]