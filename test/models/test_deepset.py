# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import jax
import jax.numpy as jnp

import netket as nk
import netket.nn as nknn


def test_deepset():
    """Test the permutation invariance"""
    L = (1.0, 1.0)
    n_particles = 6
    hilb = nk.hilbert.Particle(N=n_particles, L=L, pbc=True)
    sdim = len(hilb.extent)
    key = jax.random.PRNGKey(42)
    x = hilb.random_state(key, size=1024)
    x = x.reshape(x.shape[0], n_particles, sdim)

    xp = jnp.roll(x, 2, axis=-2)  # permute the particles

    ds = nk.models.DeepSet(
        features_phi=(16, 16), features_rho=(16, 1), squeeze_output=True
    )
    params = ds.init(key, x)
    out = ds.apply(params, x)
    outp = ds.apply(params, xp)
    assert out.shape == outp.shape
    assert out.shape == x.shape[:-2]

    assert jnp.allclose(out, outp)

    ds = nk.models.DeepSet(
        features_phi=16, features_rho=32, output_activation=nknn.gelu
    )
    params = ds.init(key, x)
    out = ds.apply(params, x)
    assert params["params"]["phi_0"]["kernel"].shape == (x.shape[-1], 16)
    assert params["params"]["rho_0"]["kernel"].shape == (16, 32)

    with pytest.raises(ValueError):
        # squeezing with dimension 3 should fail
        ds = nk.models.DeepSet(
            features_phi=(16,), features_rho=(16, 3), squeeze_output=True
        )
        params = ds.init(key, x)
        out = ds.apply(params, x)

    # flexible, should still work
    ds = nk.models.DeepSet(
        features_phi=None,
        features_rho=None,
        output_activation=None,
        hidden_activation=None,
        pooling=None,
        dtype=complex,
    )
    params = ds.init(key, x)
    out = ds.apply(params, x)


@pytest.mark.parametrize(
    "cusp_exponent", [pytest.param(None, id="cusp=None"), pytest.param(5, id="cusp=5")]
)
@pytest.mark.parametrize(
    "L",
    [
        pytest.param(1.0, id="1D"),
        pytest.param((1.0, 1.0), id="2D-Square"),
        pytest.param((1.0, 0.5), id="2D-Rectangle"),
    ],
)
def test_rel_dist_deepsets(cusp_exponent, L):

    hilb = nk.hilbert.Particle(N=2, L=L, pbc=True)
    sdim = len(hilb.extent)
    x = jnp.hstack([jnp.ones(4), -jnp.ones(4)]).reshape(1, -1)
    xp = jnp.roll(x, sdim)
    ds = nk.models.DeepSetRelDistance(
        hilbert=hilb,
        cusp_exponent=cusp_exponent,
        layers_phi=2,
        layers_rho=2,
        features_phi=(10, 10),
        features_rho=(10, 1),
    )
    p = ds.init(jax.random.PRNGKey(42), x)

    assert jnp.allclose(ds.apply(p, x), ds.apply(p, xp))


def test_rel_dist_deepsets_error():
    hilb = nk.hilbert.Particle(N=2, L=1.0, pbc=True)
    sdim = len(hilb.extent)

    x = jnp.hstack([jnp.ones(4), -jnp.ones(4)]).reshape(1, -1)
    xp = jnp.roll(x, sdim)
    ds = nk.models.DeepSetRelDistance(
        hilbert=hilb,
        layers_phi=3,
        layers_rho=3,
        features_phi=(10, 10),
        features_rho=(10, 1),
    )
    with pytest.raises(ValueError):
        p = ds.init(jax.random.PRNGKey(42), x)

    with pytest.raises(AssertionError):
        ds = nk.models.DeepSetRelDistance(
            hilbert=hilb,
            layers_phi=2,
            layers_rho=2,
            features_phi=(10, 10),
            features_rho=(10, 2),
        )
        p = ds.init(jax.random.PRNGKey(42), x)

    with pytest.raises(ValueError):
        ds = nk.models.DeepSetRelDistance(
            hilbert=nk.hilbert.Particle(N=2, L=1.0, pbc=False),
            layers_phi=2,
            layers_rho=2,
            features_phi=(10, 10),
            features_rho=(10, 2),
        )
        p = ds.init(jax.random.PRNGKey(42), x)
