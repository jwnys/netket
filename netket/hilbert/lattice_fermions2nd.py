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


class SpinlessLatticeFermions2nd(DiscreteHilbert):
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
    def size(self): 
        return self.n_fermions

    def __repr__(self):
        return "SpinlessLatticeFermions2nd(n_fermions={}, n_orbitals={})".format(
            self.n_fermions, self.n_orbitals
        )

    @property
    def _attrs(self):
        return (self.size, self.n_orbitals)
    
    @property
    def states(self):
        
        return self.n_orbitals
    

    def combinations(seq,length):
    
        if length == 0:
            return [[]]
    
        comb = []
        for i in range(len(seq)):        
            curr = seq[i]
            rem = seq[i+1:]
            
            for j in combinations(rem,length-1):
                comb.append([curr]+j)
        return comb
    

    def configurations(n_orbitals, n_fermions):
        for idx in combinations(range(n_orbitals), n_fermions):
            state = np.zeros(n_orbitals)
            for i in idx:
                state[i] = 1

            yield state

    def all_states(configs):
        return np.array(list(configs))
