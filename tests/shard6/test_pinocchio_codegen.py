"""
Compiled model functions on the Pinocchio backend.

Pinocchio is the simpler of the two backends for this: it builds SX functions natively, so
the expand step before code generation is a no-op and the generated C is compact from the
start. What still has to be checked is that the compiled function is numerically identical
and carries the derivatives the solver needs.
"""

import importlib.util
import shutil

import numpy as np
import pytest

HAS_PINOCCHIO = importlib.util.find_spec("pinocchio") is not None

pytestmark = pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio is not installed")

from bioptim import CodegenOptions, PinocchioModel  # noqa: E402
from ..utils import TestUtils  # noqa: E402


def _model_path():
    return TestUtils.bioptim_folder() + "/examples/models/pendulum.urdf"


@pytest.fixture
def cache_dir(tmp_path):
    path = tmp_path / "codegen"
    yield str(path)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def warm_model(cache_dir):
    """A model whose forward_dynamics and markers have already been compiled."""
    options = CodegenOptions(cache_dir=cache_dir, opt_level=0)
    cold = PinocchioModel(_model_path(), codegen=options)
    cold.forward_dynamics()
    cold.markers()
    cold.codegen_flush()
    return PinocchioModel(_model_path(), codegen=options)


def test_cold_model_is_sx_and_records():
    """Pinocchio builds SX natively, so nothing needs expanding before code generation."""
    model = PinocchioModel(_model_path(), codegen=False)
    assert model.forward_dynamics().class_name() == "SXFunction"


def test_warm_model_loads_compiled_functions(warm_model):
    assert warm_model.forward_dynamics().class_name() == "External"
    assert warm_model.markers().class_name() == "External"


def test_warm_model_records_nothing_more(warm_model):
    warm_model.forward_dynamics()
    warm_model.markers()
    assert not warm_model._codegen.has_pending


def test_compiled_forward_dynamics_matches(warm_model):
    reference = PinocchioModel(_model_path())
    q = np.linspace(0.1, 0.4, reference.nb_q)
    qdot = np.linspace(-0.2, 0.3, reference.nb_qdot)
    tau = np.linspace(0.5, 0.9, reference.nb_tau)
    empty = np.zeros(0)

    expected = np.array(reference.forward_dynamics()(q, qdot, tau, empty, empty)).ravel()
    actual = np.array(warm_model.forward_dynamics()(q, qdot, tau, empty, empty)).ravel()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_compiled_markers_match(warm_model):
    reference = PinocchioModel(_model_path())
    q = np.linspace(0.1, 0.4, reference.nb_q)
    empty = np.zeros(0)

    expected = np.array(reference.markers()(q, empty))
    actual = np.array(warm_model.markers()(q, empty))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_compiled_function_carries_its_derivatives(warm_model):
    """
    Without jac_<name> in the library CasADi cannot know the external's Jacobian sparsity
    and assumes it dense, which is what made a previous attempt at this unusably slow.
    """
    from casadi import Function, MX, jacobian, vertcat

    reference = PinocchioModel(_model_path())
    q = MX.sym("q", reference.nb_q)
    qdot = MX.sym("qdot", reference.nb_qdot)
    tau = MX.sym("tau", reference.nb_tau)
    empty = MX(0, 1)
    inputs = vertcat(q, qdot, tau)

    compiled_jac = jacobian(warm_model.forward_dynamics()(q, qdot, tau, empty, empty), inputs)
    reference_jac = jacobian(reference.forward_dynamics()(q, qdot, tau, empty, empty), inputs)

    assert compiled_jac.nnz() <= reference_jac.nnz()

    point = (
        np.linspace(0.1, 0.4, reference.nb_q),
        np.linspace(-0.2, 0.3, reference.nb_qdot),
        np.linspace(0.5, 0.9, reference.nb_tau),
    )
    actual = np.array(Function("a", [q, qdot, tau], [compiled_jac])(*point))
    expected = np.array(Function("e", [q, qdot, tau], [reference_jac])(*point))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_compiled_function_has_a_second_derivative(warm_model):
    """IPOPT defaults to hessian_approximation='exact'."""
    from casadi import MX, hessian, sum1, vertcat

    q = MX.sym("q", warm_model.nb_q)
    qdot = MX.sym("qdot", warm_model.nb_qdot)
    tau = MX.sym("tau", warm_model.nb_tau)
    out = warm_model.forward_dynamics()(q, qdot, tau, MX(0, 1), MX(0, 1))
    h, _ = hessian(sum1(out), vertcat(q, qdot, tau))
    assert h.nnz() > 0


def test_key_separates_pinocchio_from_biorbd(cache_dir):
    """The two backends produce different code for the same quantity; they must not share."""
    from bioptim import BiorbdModel
    from bioptim.models.codegen import compute_fast_key

    options = CodegenOptions(cache_dir=cache_dir)
    biorbd_path = TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod"
    assert compute_fast_key(PinocchioModel(_model_path()), options) != compute_fast_key(
        BiorbdModel(biorbd_path), options
    )
