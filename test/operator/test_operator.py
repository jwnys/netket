import netket as nk
from netket.operator.boson import destroy
import numpy as np
import netket.experimental as nkx

import pytest
import jax

operators = {}

# Ising 1D
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
operators["Ising 1D"] = nk.operator.Ising(hi, g, h=1.321)

# Heisenberg 1D
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
operators["Heisenberg 1D"] = nk.operator.Heisenberg(hilbert=hi, graph=g)

# Bose Hubbard
g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
hi = nk.hilbert.Fock(n_max=3, n_particles=6, N=g.n_nodes)
operators["Bose Hubbard"] = nk.operator.BoseHubbard(U=4.0, hilbert=hi, graph=g)

# Graph Hamiltonian
N = 10
sigmax = np.asarray([[0, 1], [1, 0]])
mszsz = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
edges = [[i, i + 1] for i in range(N - 1)] + [[N - 1, 0]]

g = nk.graph.Graph(edges=edges)
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)
operators["Graph Hamiltonian"] = nk.operator.GraphOperator(
    hi, g, site_ops=[sigmax], bond_ops=[mszsz]
)

g_sub = nk.graph.Graph(edges=edges[:3])  # edges of first four sites
operators["Graph Hamiltonian (on subspace)"] = nk.operator.GraphOperator(
    hi,
    g_sub,
    site_ops=[sigmax],
    bond_ops=[mszsz],
    acting_on_subspace=4,
)

# Graph Hamiltonian with colored edges
edges_c = [(i, j, i % 2) for i, j in edges]
g = nk.graph.Graph(edges=edges_c)
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)
operators["Graph Hamiltonian (colored edges)"] = nk.operator.GraphOperator(
    hi,
    g,
    site_ops=[sigmax],
    bond_ops=[1.0 * mszsz, 2.0 * mszsz],
    bond_ops_colors=[0, 1],
)

# Heisenberg with colored edges
operators["Heisenberg (colored edges)"] = nk.operator.Heisenberg(
    hi, g, J=[1, 2], sign_rule=[True, False]
)

# Custom Hamiltonian
sx = [[0, 1], [1, 0]]
sy = [[0, -1.0j], [1.0j, 0]]
sz = [[1, 0], [0, -1]]
g = nk.graph.Graph(edges=[[i, i + 1] for i in range(20)])
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)


def _loc(*args):
    return nk.operator.LocalOperator(hi, *args)


sx_hat = _loc([sx] * 3, [[0], [1], [5]])
sy_hat = _loc([sy] * 4, [[2], [3], [4], [9]])
szsz_hat = _loc(sz, [0]) @ _loc(sz, [1])
szsz_hat += _loc(sz, [4]) @ _loc(sz, [5])
szsz_hat += _loc(sz, [6]) @ _loc(sz, [8])
szsz_hat += _loc(sz, [7]) @ _loc(sz, [0])

operators["Custom Hamiltonian"] = sx_hat + sy_hat + szsz_hat
operators["Custom Hamiltonian Prod"] = sx_hat * 1.5 + (2.0 * sy_hat)

operators["Pauli Hamiltonian (XX)"] = nk.operator.PauliStrings(["XX"], [0.1])
operators["Pauli Hamiltonian (XX+YZ+IZ)"] = nk.operator.PauliStrings(
    ["XX", "YZ", "IZ"], [0.1, 0.2, -1.4]
)

hi = nkx.hilbert.SpinOrbitalFermions(5)
operators["FermionOperator2nd"] = nkx.operator.FermionOperator2nd(
    hi,
    terms=(((0, 1), (3, 0)), ((3, 1), (0, 0))),
    weights=(0.5 + 0.3j, 0.5 - 0.3j),  # must add h.c.
)

op_special = {}
for name, op in operators.items():
    if hasattr(op, "to_local_operator"):
        op_special[name] = op


@pytest.mark.parametrize("attr", ["get_conn", "get_conn_padded"])
@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_produce_elements_in_hilbert(op, attr):
    rng = nk.jax.PRNGSeq(0)
    hi = op.hilbert
    get_conn_fun = getattr(op, attr)

    assert len(hi.local_states) == hi.local_size
    assert hi.size > 0

    local_states = hi.local_states
    max_conn_size = op.max_conn_size
    rstates = hi.random_state(rng.next(), 1000)

    for i in range(len(rstates)):
        rstatet, mels = get_conn_fun(rstates[i])

        assert np.all(np.isin(rstatet, local_states))
        assert len(mels) <= max_conn_size


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_is_hermitean(op):
    rng = nk.jax.PRNGSeq(20)

    hi = op.hilbert
    assert len(hi.local_states) == hi.local_size

    rstates = hi.random_state(rng.next(), 100)
    for i in range(len(rstates)):
        rstate = rstates[i]
        rstatet, mels = op.get_conn(rstate)

        for k, state in enumerate(rstatet):

            invstates, mels1 = op.get_conn(state)

            found = False
            for kp, invstate in enumerate(invstates):
                if np.array_equal(rstate, invstate.flatten()):
                    found = True
                    assert mels1[kp] == np.conj(mels[k])
                    break

            assert found


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_repr(op):
    assert type(op).__name__ in repr(op)


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_get_conn_numpy_closure(op):
    hi = op.hilbert
    closure = op._get_conn_flattened_closure()
    v = hi.random_state(jax.random.PRNGKey(0), 120)
    conn = np.empty(v.shape[0], dtype=np.intp)

    vp, mels = closure(np.asarray(v), conn)
    vp2, mels2 = op.get_conn_flattened(v, conn, pad=False)

    np.testing.assert_equal(vp, vp2)
    np.testing.assert_equal(mels, mels2)


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(s, id=f"shape={s}")
        for s in [
            (2,),
            (
                2,
                1,
            ),
            (2, 1, 1),
        ]
    ],
)
def test_get_conn_padded(op, shape):
    hi = op.hilbert

    v = hi.random_state(jax.random.PRNGKey(0), shape)

    vp, mels = op.get_conn_padded(v)

    assert vp.ndim == v.ndim + 1
    assert mels.ndim == v.ndim

    vp_f, mels_f = op.get_conn_padded(v.reshape(-1, hi.size))
    np.testing.assert_allclose(vp_f, vp.reshape(-1, *vp.shape[-2:]))
    np.testing.assert_allclose(mels_f, mels.reshape(-1, mels.shape[-1]))


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in op_special.items()]
)
def test_to_local_operator(op):
    op.to_local_operator()
    # TODO check dense representaiton.


def test_no_segfault():
    g = nk.graph.Hypercube(8, 1)
    hi = nk.hilbert.Spin(0.5, N=g.n_nodes)

    lo = nk.operator.LocalOperator(hi, [[1, 0], [0, 1]], [0])
    lo = lo.transpose()

    hi = None

    lo = lo @ lo

    assert True


def test_deduced_hilbert_pauli():
    op = nk.operator.PauliStrings(["XXI", "YZX", "IZX"], [0.1, 0.2, -1.4])
    assert op.hilbert.size == 3
    assert len(op.hilbert.local_states) == 2
    assert isinstance(op.hilbert, nk.hilbert.Qubit)
    assert np.allclose(op.hilbert.local_states, (0, 1))


def test_Heisenberg():
    g = nk.graph.Hypercube(8, 1)
    hi = nk.hilbert.Spin(0.5) ** 8

    def gs_energy(ham):
        return nk.exact.lanczos_ed(ham)

    ha1 = nk.operator.Heisenberg(hi, graph=g)
    ha2 = nk.operator.Heisenberg(hi, graph=g, J=2.0)

    assert 2 * gs_energy(ha1) == pytest.approx(gs_energy(ha2))

    ha1 = nk.operator.Heisenberg(hi, graph=g, sign_rule=True)
    ha2 = nk.operator.Heisenberg(hi, graph=g, sign_rule=False)

    assert gs_energy(ha1) == pytest.approx(gs_energy(ha2))

    with pytest.raises(
        ValueError, match=r"sign_rule=True specified for a non-bipartite lattice"
    ):
        g = nk.graph.Hypercube(7, 1)
        hi = nk.hilbert.Spin(0.5, N=g.n_nodes)

        assert not g.is_bipartite()

        nk.operator.Heisenberg(hi, graph=g, sign_rule=True)

    L = 8
    edges = [(i, (i + 1) % L, 0) for i in range(L)] + [
        (i, (i + 2) % L, 1) for i in range(L)
    ]
    hi = nk.hilbert.Spin(0.5) ** L
    g = nk.graph.Graph(edges=edges)
    ha1 = nk.operator.Heisenberg(hi, graph=g, J=[1, 0.5])
    ha2 = nk.operator.Heisenberg(hi, graph=g, J=[1, 0.5], sign_rule=[True, False])

    assert gs_energy(ha1) == pytest.approx(gs_energy(ha2))


@pytest.mark.parametrize(
    "hilbert",
    [
        pytest.param(hi, id=str(hi))
        for hi in (nk.hilbert.Spin(1 / 2, 2), nk.hilbert.Qubit(2), None)
    ],
)
def test_pauli(hilbert):
    operators = ["XX", "YZ", "IZ"]
    weights = [0.1, 0.2, -1.4]

    if hilbert is None:
        op = nk.operator.PauliStrings(operators, weights)
    else:
        op = nk.operator.PauliStrings(hilbert, operators, weights)
        assert op.hilbert == hilbert

    op_l = (
        0.1
        * nk.operator.spin.sigmax(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmax(op.hilbert, 1)
    )
    op_l += (
        0.2
        * nk.operator.spin.sigmay(op.hilbert, 0)
        * nk.operator.spin.sigmaz(op.hilbert, 1)
    )
    op_l -= 1.4 * nk.operator.spin.sigmaz(op.hilbert, 1)

    assert np.allclose(op.to_dense(), op_l.to_dense())

    assert op.to_sparse().shape == op_l.to_sparse().shape


def test_pauli_trivials():
    operators = ["XX", "YZ", "IZ"]
    weights = [0.1, 0.2, -1.4]

    # without weight
    nk.operator.PauliStrings(operators)
    nk.operator.PauliStrings(nk.hilbert.Qubit(2), operators)
    nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 2), operators)

    # using keywords
    nk.operator.PauliStrings(operators, weights)
    nk.operator.PauliStrings(nk.hilbert.Qubit(2), operators, weights)
    nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 2), operators, weights)

    nk.operator.PauliStrings.identity(nk.hilbert.Qubit(2))
    nk.operator.PauliStrings.identity(nk.hilbert.Spin(1 / 2, 2))


def test_pauli_cutoff():
    weights = [1, -1, 1]
    operators = ["ZI", "IZ", "XX"]
    op = nk.operator.PauliStrings(operators, weights, cutoff=1e-8)
    hilbert = op.hilbert
    x = np.ones((2,)) * hilbert.local_states[0]
    xp, mels = op.get_conn(x)
    assert xp.shape[-1] == hilbert.size
    assert xp.shape[-2] == 1


def test_pauli_order():
    """Check related to PR #836"""
    coeff1 = 1 + 0.9j
    coeff2 = 0.3 + 0.43j
    op = nk.operator.PauliStrings(["IZXY", "ZZYX"], [coeff1, coeff2])
    op1 = nk.operator.PauliStrings(["IZXY"], [coeff1])
    op2 = nk.operator.PauliStrings(["ZZYX"], [coeff2])
    op1_true = (
        coeff1
        * nk.operator.spin.sigmaz(op.hilbert, 1, dtype=complex)
        * nk.operator.spin.sigmax(op.hilbert, 2)
        * nk.operator.spin.sigmay(op.hilbert, 3)
    )
    op2_true = (
        coeff2
        * nk.operator.spin.sigmaz(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmaz(op.hilbert, 1)
        * nk.operator.spin.sigmay(op.hilbert, 2)
        * nk.operator.spin.sigmax(op.hilbert, 3)
    )
    assert np.allclose(op1.to_dense(), op1_true.to_dense())
    assert np.allclose(op2.to_dense(), op2_true.to_dense())
    assert np.allclose(op.to_dense(), (op1_true.to_dense() + op2_true.to_dense()))

    v = op.hilbert.all_states()
    vp, mels = op.get_conn_padded(v)
    assert vp.shape[1] == 1
    assert mels.shape[1] == 1


def test_pauli_matmul():
    op1 = nk.operator.PauliStrings(["X"], [1])
    op2 = nk.operator.PauliStrings(["Y", "Z"], [1, 1])
    op_true_mm = nk.operator.PauliStrings(["Z", "Y"], [1j, -1j])
    op_mm = op1 @ op2
    assert np.allclose(op_mm.to_dense(), op_true_mm.to_dense())

    # more extensive test
    operators1, weights1 = ["XII", "IXY"], [1, 3]
    op1 = nk.operator.PauliStrings(operators1, weights1)
    operators2, weights2 = ["XZZ", "YIZ", "ZII", "IIY"], [1, 0.2, 0.3, 3.1]
    op2 = nk.operator.PauliStrings(operators2, weights2)
    op = op1 @ op2
    op1_true = weights1[0] * nk.operator.spin.sigmax(op.hilbert, 0, dtype=complex)
    op1_true += (
        weights1[1]
        * nk.operator.spin.sigmax(op.hilbert, 1, dtype=complex)
        * nk.operator.spin.sigmay(op.hilbert, 2)
    )
    op2_true = (
        weights2[0]
        * nk.operator.spin.sigmax(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmaz(op.hilbert, 1)
        * nk.operator.spin.sigmaz(op.hilbert, 2)
    )
    op2_true += (
        weights2[1]
        * nk.operator.spin.sigmay(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmaz(op.hilbert, 2)
    )
    op2_true += weights2[2] * nk.operator.spin.sigmaz(op.hilbert, 0, dtype=complex)
    op2_true += weights2[3] * nk.operator.spin.sigmay(op.hilbert, 2, dtype=complex)
    assert np.allclose((op1_true @ op2_true).to_dense(), op.to_dense())


def test_pauli_add_and_multiply():
    op1 = nk.operator.PauliStrings(["X"], [1])
    op2 = nk.operator.PauliStrings(["X", "Y", "Z"], [-1, 1, 1])
    op_true_add = nk.operator.PauliStrings(["Y", "Z"], [1, 1])
    op_add = op1 + op2
    assert np.allclose(op_add.to_dense(), op_true_add.to_dense())
    op_true_multiply = nk.operator.PauliStrings(["X", "Y", "Z"], [-2, 2, 2])
    op_multiply = op2 * 2  # right
    assert np.allclose(op_multiply.to_dense(), op_true_multiply.to_dense())
    op_multiply = 2 * op2  # left
    assert np.allclose(op_multiply.to_dense(), op_true_multiply.to_dense())

    op_add_cte = nk.operator.PauliStrings(["X", "Y", "Z"], [-1, 1, 1]) + 2
    op_true_add_cte = nk.operator.PauliStrings(["X", "Y", "Z", "I"], [-1, 1, 1, 2])
    assert np.allclose(op_add_cte.to_dense(), op_true_add_cte.to_dense())


@pytest.mark.parametrize(
    "hilbert",
    [
        pytest.param(hi, id=str(hi))
        for hi in (nk.hilbert.Spin(1 / 2, 2), nk.hilbert.Qubit(2), None)
    ],
)
def test_pauli_output(hilbert):
    ha = nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 2), ["IZ", "ZI"], [1.0, 1.0])
    all_states = ha.hilbert.all_states()
    xp, _ = ha.get_conn_padded(all_states)
    xp = xp.reshape(-1, ha.hilbert.size)

    # following will throw an error if the output is not a valid hilbert state
    for xpi in xp:
        assert np.any(xpi == all_states), "{} not in hilbert space {}".format(
            xpi, ha.hilbert
        )


def test_pauli_dense():

    for op in ("I", "X", "Y", "Z"):
        ha1 = nk.operator.PauliStrings(nk.hilbert.Qubit(1), [op], [1])
        ha2 = nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 1), [op], [1])
        assert np.allclose(ha1.to_dense(), ha2.to_dense())


def test_pauli_zero():
    op1 = nk.operator.PauliStrings(["IZ"], [1])
    op2 = nk.operator.PauliStrings(["IZ"], [-1])
    op = op1 + op2

    all_states = op.hilbert.all_states()
    _, mels = op.get_conn_padded(all_states)
    assert np.allclose(mels, 0)


def test_operator_on_subspace():
    hi = nk.hilbert.Spin(1 / 2, N=3) * nk.hilbert.Qubit(N=3)
    g = nk.graph.Chain(3, pbc=False)

    h1 = nk.operator.GraphOperator(hi, g, bond_ops=[mszsz], acting_on_subspace=0)
    assert h1.acting_on_subspace == list(range(3))
    assert nk.exact.lanczos_ed(h1)[0] == pytest.approx(-2.0)

    h2 = nk.operator.GraphOperator(hi, g, bond_ops=[mszsz], acting_on_subspace=3)
    assert h2.acting_on_subspace == list(range(3, 6))
    assert nk.exact.lanczos_ed(h2)[0] == pytest.approx(-2.0)

    h12 = h1 + h2
    assert sorted(h12.acting_on) == [[0, 1], [1, 2], [3, 4], [4, 5]]
    assert nk.exact.lanczos_ed(h12)[0] == pytest.approx(-4.0)

    h3 = nk.operator.GraphOperator(
        hi, g, bond_ops=[mszsz], acting_on_subspace=[0, 2, 4]
    )
    assert h3.acting_on_subspace == [0, 2, 4]
    assert nk.exact.lanczos_ed(h3)[0] == pytest.approx(-2.0)
    assert h3.acting_on == [[0, 2], [2, 4]]

    h4 = nk.operator.Heisenberg(hi, g, acting_on_subspace=0)
    assert h4.acting_on == h1.acting_on


op_ferm = {}
hi = nkx.hilbert.SpinOrbitalFermions(3)
op_ferm["FermionOperator2nd_hermitian"] = (
    nkx.operator.FermionOperator2nd(
        hi, terms=(((0, 0), (1, 1)), ((1, 0), (0, 1))), weights=(1.0 + 1j, 1 - 1j)
    ),
    True,
)
op_ferm["FermionOperator2nd_not_hermitian"] = (
    nkx.operator.FermionOperator2nd(
        hi, terms=(((0, 0), (2, 1)), ((1, 0), (0, 1))), weights=(1.0 + 1j, 1 - 1j)
    ),
    False,
)

op_ferm["FermionOperator2nd_hermitian_3term"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        (((0, 0), (1, 1), (2, 1)), ((2, 0), (1, 0), (0, 1))),
        weights=(1.0 - 1j, 1 + 1j),
    ),
    True,
)
op_ferm["FermionOperator2nd_not_hermitian_3term"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        (((0, 0), (1, 1), (2, 1)), ((3, 0), (1, 0), (0, 1))),
        weights=(1.0 - 1j, 2 + 2j),
    ),
    False,
)

op_ferm["fermihubbard_int"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        terms=(
            ((0, 1), (0, 0), (1, 1), (1, 0)),
            ((0, 1), (0, 0), (1, 1), (1, 0)),
            ((0, 1), (0, 0), (1, 1), (1, 0)),
            ((0, 1), (0, 0), (1, 1), (1, 0)),
        ),
        weights=(1.0, 1.0, 1.0, 1.0),
    ),
    True,
)

op_ferm["ordering"] = (
    nkx.operator.FermionOperator2nd(
        hi,
        terms=(((0, 1), (0, 0), (1, 1), (1, 0)), ((1, 1), (1, 0), (0, 1), (0, 0))),
        weights=(1.0 - 1j, 1 + 1j),
    ),
    True,
)


@pytest.mark.parametrize(
    "op_ferm, is_hermitian",
    [pytest.param(op, is_herm, id=name) for name, (op, is_herm) in op_ferm.items()],
)
def test_is_hermitian_fermion2nd(op_ferm, is_hermitian):
    print("OPERATOR", op_ferm._operators)
    assert op_ferm.is_hermitian == is_hermitian


def test_openfermion_conversion():
    # skip test if openfermion not installed
    pytest.importorskip("openfermion")
    from openfermion.ops import QubitOperator, FermionOperator

    # first term is a constant
    of_qubit_operator = (
        QubitOperator("") + 0.5 * QubitOperator("X0 X3") + 0.3 * QubitOperator("Z0")
    )

    # no extra info given
    ps = nk.operator.PauliStrings.from_openfermion(of_qubit_operator)
    assert isinstance(ps, nk.operator.PauliStrings)
    assert isinstance(ps.hilbert, nk.hilbert.Qubit)
    assert ps.hilbert.size == 4

    # number of qubits given
    ps = nk.operator.PauliStrings.from_openfermion(of_qubit_operator, n_qubits=6)
    assert isinstance(ps, nk.operator.PauliStrings)
    # check default
    assert isinstance(ps.hilbert, nk.hilbert.Qubit)
    assert ps.hilbert.size == 6

    # with hilbert
    hilbert = nk.hilbert.Spin(1 / 2, 6)
    ps = nk.operator.PauliStrings.from_openfermion(hilbert, of_qubit_operator)
    assert ps.hilbert == hilbert
    assert ps.hilbert.size == 6

    # FermionOperator
    of_fermion_operator = (
        FermionOperator("")  # todo
        + FermionOperator("0^ 3", 0.5 + 0.3j)
        + FermionOperator("3^ 0", 0.5 - 0.3j)
    )

    # no extra info given
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(of_fermion_operator)
    assert fo2.hilbert.size == 4

    # number of orbitals given
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(
        of_fermion_operator, n_orbitals=4
    )
    assert isinstance(fo2, nkx.operator.FermionOperator2nd)
    assert isinstance(fo2.hilbert, nkx.hilbert.SpinOrbitalFermions)
    assert fo2.hilbert.size == 4

    # with hilbert
    hilbert = nkx.hilbert.SpinOrbitalFermions(6)
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(hilbert, of_fermion_operator)
    assert fo2.hilbert == hilbert
    assert fo2.hilbert.size == 6

    # to check that the constraints are met (convention wrt ordering of states with different spin)
    from openfermion.hamiltonians import fermi_hubbard

    hilbert = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2, n_fermions=(2, 1))
    of_fermion_operator = fermi_hubbard(1, 3, tunneling=1, coulomb=0, spinless=False)
    fo2 = nkx.operator.FermionOperator2nd.from_openfermion(
        hilbert, of_fermion_operator, convert_spin_blocks=True
    )
    assert fo2.hilbert.size == 3 * 2
    # will fail of we go outside of the allowed states with openfermion operators
    fo2.to_dense()


def test_fermion_operator_with_strings():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    terms = (((0, 1), (2, 0)),)
    op1 = nkx.operator.FermionOperator2nd(hi, terms)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 2",))
    assert np.allclose(op1.to_dense(), op2.to_dense())

    terms = (((0, 1), (1, 0)), ((2, 1), (1, 0)))
    weights = (0.5 - 0.5j, 0.5 + 0.5j)
    op1 = nkx.operator.FermionOperator2nd(hi, terms, weights)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 1", "2^ 1"), weights)
    assert np.allclose(op1.to_dense(), op2.to_dense())

    terms = (((0, 1), (1, 0), (2, 1)), ((2, 1), (1, 0), (0, 1)))
    weights = (0.5 - 0.5j, 0.5 + 0.5j)
    op1 = nkx.operator.FermionOperator2nd(hi, terms, weights)
    op2 = nkx.operator.FermionOperator2nd(hi, ("0^ 1 2^", "2^ 1 0^"), weights)
    assert np.allclose(op1.to_dense(), op2.to_dense())


def compare_openfermion_fermions():
    # skip test if openfermion not installed
    pytest.importorskip("openfermion")
    from openfermion import FermionOperator, get_sparse_operator

    # openfermion
    of = FermionOperator("0^ 1", 1.0) + FermionOperator("1^ 0", 1.0)
    of_dense = get_sparse_operator(of).todense()
    # from_openfermion
    fo = nkx.operator.FermionOperator2nd.from_openfermion(of)
    fo_dense = fo.to_dense()
    # FermionOperator2nd
    hi = nkx.hilbert.SpinOrbitalFermions(2)  # two sites
    fermop = nkx.operator.FermionOperator2nd(
        hi, terms=(((0, 1), (1, 0)), ((1, 1), (0, 0))), weights=(1.0, 1.0)
    )
    fermop_dense = fermop.to_dense()
    # compare openfermion vs from_openfermion
    assert np.array_equal(of_dense, fo_dense)
    # compare openfermion vs FermionOperator2nd
    assert np.array_equal(of_dense, fermop_dense)
    # compare from_openfermion vs FermionOperator 2nd
    assert np.array_equal(fo_dense, fermop_dense)


def test_add_fermions():
    hi = nkx.hilbert.SpinOrbitalFermions(5)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("1^ 2"), weights=(1,), constant=2)
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4"), weights=(1.3,), constant=5.7
    )
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(1.3, 1), constant=7.7
    )
    op4 = op3 * 2
    op5 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(2 * 1.3, 2 * 1), constant=2 * 7.7
    )
    assert np.allclose((op1 + op2).to_dense(), op3.to_dense())
    assert np.allclose(op4.to_dense(), op5.to_dense())


def test_create_annihil_number():
    hi = nkx.hilbert.SpinOrbitalFermions(5)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2))

    from netket.experimental.operator.fermion import destroy, create, number

    def c(site):
        return destroy(hi, site)

    def cdag(site):
        return create(hi, site)

    def cn(site):
        return number(hi, site)

    op2 = 0.3 * cn(0) + 2 * cdag(1) * c(2)
    assert np.allclose(op1.to_dense(), op2.to_dense())
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 1", "1^ 2"), weights=(1 + 1j, 2 - 2j), constant=2
    )
    op4 = (1 + 1j) * cdag(0) * c(1) + (2 - 2j) * cdag(1) * c(2) + 2
    assert np.allclose(op3.to_dense(), op4.to_dense())


def test_operations_fermions():
    hi = nkx.hilbert.SpinOrbitalFermions(5)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("1^ 2",), weights=(1,), constant=2)
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4"), weights=(1.3,), constant=5.7
    )
    op2copy = op2.copy()
    assert op2copy.hilbert == op2.hilbert
    assert np.allclose(list(op2._operators.keys()), list(op2copy._operators.keys()))
    assert np.allclose(list(op2._operators.values()), list(op2copy._operators.values()))
    assert op2.is_hermitian == op2copy.is_hermitian
    assert np.allclose(op2.to_dense(), op2copy.to_dense())

    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(1.3, 1), constant=7.7
    )
    op12 = op1.copy()
    op12 += op2
    assert np.allclose((op1 + op2).to_dense(), op3.to_dense())
    assert np.allclose(op12.to_dense(), op3.to_dense())

    op4 = op3 * 2
    op5 = nkx.operator.FermionOperator2nd(
        hi, terms=("3^ 4", "1^ 2"), weights=(2 * 1.3, 2 * 1), constant=2 * 7.7
    )
    op4b = op3.copy()
    op4b *= 2
    assert np.allclose(op4.to_dense(), op5.to_dense())
    assert np.allclose(op4b.to_dense(), op5.to_dense())

    op6 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 2", "0^ 1"), weights=(1j, -1.0j), constant=7.7
    )
    op7 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 2", "0^ 1"), weights=(1, 1), constant=7.7
    )
    op8 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 2", "0^ 1"), weights=(1.0 + 1j, 1 - 1j), constant=2 * 7.7
    )
    op67 = op6.copy()
    op67 += op7
    assert np.allclose((op6 + op7).to_dense(), op8.to_dense())
    assert np.allclose(op67.to_dense(), op8.to_dense())


def test_fermion_op_matmul():
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1 = nkx.operator.FermionOperator2nd(hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2))

    # multiply with a real constant
    op_real = nkx.operator.FermionOperator2nd(hi, [], [], constant=2.0)
    assert np.allclose((op1 @ op_real).to_dense(), (op1 * 2).to_dense())
    assert np.allclose((op1 * op_real).to_dense(), (op1 * 2).to_dense())

    # multiply with a real+complex constant
    op_complex = nkx.operator.FermionOperator2nd(hi, [], [], constant=2.0 + 2j)
    assert np.allclose((op1 @ op_complex).to_dense(), (op1 * (2 + 2j)).to_dense())
    assert np.allclose((op1 * op_complex).to_dense(), (op1 * (2 + 2j)).to_dense())

    # multiply with another operator
    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 1", "0^ 2"), weights=(1 + 1j, 0.5)
    )
    op2b = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 0 1^ 1", "0^ 0 0^ 2", "1^ 2 1^ 1", "1^ 2 0^ 2"),
        weights=(0.3 * (1 + 1j), 0.3 * 0.5, 2 * (1 + 1j), 2 * 0.5),
    )
    assert np.allclose(
        (op1 @ op2).to_dense(),
        op2b.to_dense(),
    )
    assert np.allclose(
        (op1 * op2).to_dense(),
        op2b.to_dense(),
    )

    # multiply with another operator + constant
    op3 = nkx.operator.FermionOperator2nd(
        hi, terms=("1^ 1",), weights=(1 + 1j,), constant=5
    )
    op3b = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 0 1^ 1", "0^ 0", "1^ 2 1^ 1", "1^ 2"),
        weights=(0.3 * (1 + 1j), 5 * 0.3, 2 * (1 + 1j), 10),
        constant=0,
    )
    assert np.allclose(
        (op1 @ op3).to_dense(),
        op3b.to_dense(),
    )
    assert np.allclose(
        (op1 * op3).to_dense(),
        op3b.to_dense(),
    )


def test_fermion_add_sub_mul():
    # check addition
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    op1 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2"), weights=(0.3, 2), constant=2
    )
    assert np.allclose((op1 + op1).to_dense(), 2 * op1.to_dense())

    op2 = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "0^ 1"), weights=(0.5, 4j), constant=1
    )
    op2b = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2", "0^ 1"), weights=(0.3 + 0.5, 2, 4j), constant=3
    )
    assert np.allclose((op1 + op2).to_dense(), op2b.to_dense())
    op2c = op2.copy()
    op2c += op1
    assert np.allclose(op2c.to_dense(), op2b.to_dense())

    # check substraction
    op2d = nkx.operator.FermionOperator2nd(
        hi, terms=("0^ 0", "1^ 2", "0^ 1"), weights=(0.3 - 0.5, 2, -4j), constant=1
    )
    assert np.allclose((op1 - op2).to_dense(), op2d.to_dense())
    op1c = op1.copy()
    op1c -= op2
    assert np.allclose(op1c.to_dense(), op2d.to_dense())

    # check multiplication with scalar
    op1f = nkx.operator.FermionOperator2nd(
        hi,
        terms=("0^ 0", "1^ 2"),
        weights=(
            3,
            20,
        ),
        constant=20,
    )
    op1c = op1.copy()
    op1c *= 10
    assert np.allclose((op1 * 10).to_dense(), op1f.to_dense())
    assert np.allclose(op1c.to_dense(), op1f.to_dense())
