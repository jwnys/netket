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

from typing import Optional, List

from netket.graph import AbstractGraph

from netket.hilbert.discrete_hilbert import DiscreteHilbert
from netket.hilbert._deprecations import graph_to_N_depwarn

import numpy as np

import numba as nb
import jax

from functools import partial


@nb.njit
def comb(N, k):
    """
    Numba implementation of _comb_int of scipy.special.comb
    https://github.com/scipy/scipy/blob/master/scipy/special/_comb.pyx
    An even faster version can be implemented (see _comb_long)
    """
    N = int(N)
    k = int(k)

    if k > N or N < 0 or k < 0:
        return 0

    M = N + 1
    nterms = min(k, N - k)

    numerator = 1
    denominator = 1
    for j in range(1, nterms + 1):
        numerator *= M - j
        denominator *= j

    return numerator // denominator


class SpinlessLatticeFermions1st(DiscreteHilbert):
    def __init__(
        self, n_fermions: int, n_sites: int = 1, graph: Optional[AbstractGraph] = None
    ):
        r"""Hilbert space that represents spinless fermions

        Args:
            n_fermions: number of fermions
            n_sites: number of sites on the lattice
            graph: alternative to specify the number of fermions
        """
        n_sites = graph_to_N_depwarn(N=n_sites, graph=graph)
        n_local_states = 1  # spinless
        self.n_fermions = n_fermions
        self.n_orbitals = n_sites * n_local_states  # spinless
        self._shape = [self.n_orbitals] * self.n_fermions
        self.adj = None  # everything connected
        if graph is not None:
            self.adj = graph.adjacency_list()  # for sampling (List[List])

    @property
    def local_size(self) -> int:
        r"""Size of the local degrees of freedom that make the total hilbert space."""
        return self.n_orbitals

    @property
    def local_states(self):
        return np.arange(self.n_orbitals)

    @property
    def n_states(self):
        return comb(self.n_orbitals, self.n_fermions)  # , exact=True, repetition=False)

    @property
    def size(self):
        return self.n_fermions

    def __repr__(self):
        return "SpinlessLatticeFermions1st(n_fermions={}, n_orbitals={})".format(
            self.n_fermions, self.n_orbitals
        )

    @property
    def _attrs(self):
        return (self.size, self.n_orbitals)

    def states_at_index(self, i: int) -> Optional[List[float]]:
        return self.local_states

    def _states_to_numbers(self, states, out=None):
        return map_index(states, self.n_fermions, self.n_orbitals, out, is_sorted=False)

    def _numbers_to_states(self, numbers, out=None):
        return map_state(numbers, self.n_fermions, out)

    @property
    def is_finite(self) -> bool:
        return True

    def to_canonical(self, states):
        return permutation_parity(states.reshape(-1, self.n_fermions))  # xp, mels

    def occupation_number(self, states):
        return occ_num_fn(states, self.n_orbitals)

    def occupation_number_fn(self):
        return partial(occ_num_fn, n_orbitals=self.n_orbitals)


@nb.njit
def permutation_parity(permutations):
    parities = np.empty((permutations.shape[0],), dtype=np.int_)
    sorted_terms = np.empty(permutations.shape, dtype=permutations.dtype)
    for b in range(permutations.shape[0]):
        sort_idx = np.argsort(permutations[b, :])
        sorted_terms[b, :] = np.array([permutations[b, i] for i in sort_idx])
        parities[b] = _permutation_parity(sort_idx)
    return sorted_terms, parities


@nb.njit
def _permutation_parity(permutation):
    permutation = list(permutation)
    length = len(permutation)
    elements_seen = [False] * length
    cycles = 0
    for index, already_seen in enumerate(elements_seen):
        if already_seen:
            continue
        cycles += 1
        current = index
        while not elements_seen[current]:
            elements_seen[current] = True
            current = permutation[current]
    is_even = (length - cycles) % 2 == 0
    return +1 if is_even else -1


@partial(jax.jit, static_argnums=(1,))
def occ_num_fn(states, n_orbitals):
    occ = np.zeros((states.shape[0], n_orbitals), dtype=states.dtype)

    def change(o, st):
        return o.at[st].set(1)

    return jax.vmap(change)(occ, states.astype(int))


@nb.njit
def _largest_n_not_exeeding(number, k):
    """See wikipedia: Combinatorial number system"""
    n = k - 1
    c_prev = 0
    while True:
        c = comb(n, k)
        if c > number:
            return n - 1, c_prev
        n += 1
        c_prev = c


@nb.njit
def _state(number, k):
    """See wikipedia: Combinatorial number system"""
    num = number
    state = np.empty((k,), dtype=np.intp)
    for i in range(k):
        n, diff = _largest_n_not_exeeding(num, k - i)
        state[i] = n
        num -= diff
    return np.flip(state)


@nb.njit
def map_state(numbers, n_fermions, out):
    if numbers.ndim != 1:
        raise RuntimeError("Invalid input shape, expecting a 1d array.")
    for b, number in enumerate(numbers):
        out[b] = _state(number, n_fermions)
    return out


@nb.njit
def _index(locs, n_fermions, n_orbitals, is_sorted=False):
    """See wikipedia: Combinatorial number system"""
    if not is_sorted:
        locs = np.sort(locs)
    idx = 0
    for k in range(n_fermions):
        ak = locs[k]
        bk = k + 1
        idx += int(comb(ak, bk))
    return np.intp(idx)


@nb.njit
def map_index(locs_arr, n_fermions, n_orbitals, out, is_sorted=False):
    if locs_arr.ndim != 2:
        raise RuntimeError("Invalid input shape, expecting a 2d array.")
    for b in range(locs_arr.shape[0]):
        out[b] = _index(locs_arr[b, :], n_fermions, n_orbitals, is_sorted=is_sorted)
    return out
