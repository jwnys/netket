# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax import numpy as jnp

from netket.hilbert import SpinlessLatticeFermions1st
from netket.utils.dispatch import dispatch


@dispatch
def random_state(
    hilb: SpinlessLatticeFermions1st, key, batches: int, *, dtype
):  # noqa: F811
    choices = jnp.arange(hilb.n_orbitals).astype(dtype)
    shape = (hilb.n_fermions,)
    keys = jax.random.split(key, batches)
    rand_int = jax.vmap(
        lambda key: jax.random.choice(key, choices, shape=shape, replace=False)
    )
    states = rand_int(keys)
    return jnp.sort(states, axis=-1)


@dispatch
def flip_state_scalar(hilb: SpinlessLatticeFermions1st, key, state, fermion_index):
    """TODO: need to sort things again at the end!!! But this will break the tests"""
    # for each orbital, get an index of where we can find it in the state
    old_orbital = state[fermion_index]
    adj = hilb.adj  # orbitals are just the sites
    if adj is not None:  # no connectivity
        choices = adj[old_orbital]
    else:
        choices = jnp.arange(hilb.n_orbitals)
    new_orbital = jax.random.choice(key, choices)
    idx = jnp.where(state == new_orbital, size=1, fill_value=-1)[0][0]
    replace_fn = lambda _: state.at[fermion_index].set(new_orbital)
    nothing_fn = lambda _: state
    return jax.lax.cond(idx == -1, replace_fn, nothing_fn, None), old_orbital
