"""
Tests for the model-function codegen cache key and the compiled round-trip.

The key is the part of this feature that fails silently when it is wrong: a key that is too
coarse serves a compiled function that does not match the model, and nothing complains. So
most of what is checked here is that the key *moves* when it must.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from casadi import MX, Function, jacobian, vertcat

from bioptim import BiorbdModel, CodegenOptions
from bioptim.models.codegen import compute_fast_key, symbol_for
from bioptim.models.codegen.cache import ModelFunctionCache
from tests.utils import TestUtils

MODEL = TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod"
CUBE = TestUtils.bioptim_folder() + "/examples/models/cube.bioMod"


@pytest.fixture
def cache_dir(tmp_path):
    path = tmp_path / "codegen"
    yield str(path)
    shutil.rmtree(path, ignore_errors=True)


def _key(model, options):
    return compute_fast_key(model, options)


def _pendulum_ocp(codegen, use_sx: bool, expand: bool):
    """A minimal torque-driven pendulum OCP, built on a model with the given codegen setting."""
    from bioptim import (
        DynamicsOptions,
        InitialGuessList,
        Objective,
        ObjectiveFcn,
        OdeSolver,
        OptimalControlProgram,
        TorqueBiorbdModel,
    )
    from bioptim import BoundsList as _BoundsList

    model = TorqueBiorbdModel(MODEL, codegen=codegen)
    x_bounds = _BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    x_init = InitialGuessList()
    x_init["q"] = [0] * model.nb_q
    x_init["qdot"] = [0] * model.nb_qdot
    u_bounds = _BoundsList()
    u_bounds["tau"] = [-100] * model.nb_tau, [100] * model.nb_tau
    u_bounds["tau"][1, :] = 0
    u_init = InitialGuessList()
    u_init["tau"] = [0] * model.nb_tau

    return OptimalControlProgram(
        model,
        20,
        1.0,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(), expand_dynamics=expand),
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        use_sx=use_sx,
    )


def test_key_is_stable_for_the_same_model():
    """Two identical models must share a cache entry, or nothing is ever reused."""
    options = CodegenOptions()
    assert _key(BiorbdModel(MODEL), options) == _key(BiorbdModel(MODEL), options)


def test_key_changes_with_the_model_file():
    options = CodegenOptions()
    assert _key(BiorbdModel(MODEL), options) != _key(BiorbdModel(CUBE), options)


def test_key_follows_file_contents_not_the_path(tmp_path):
    """A copy of a model must hit the same entry; an edited model must not.

    The copies live next to the original because a .bioMod refers to its mesh files by a
    path relative to itself, so a copy elsewhere will not load.
    """
    options = CodegenOptions()
    original = Path(MODEL)
    copied = original.with_name("tmp_codegen_copy.bioMod")
    edited = original.with_name("tmp_codegen_edited.bioMod")
    try:
        shutil.copy(original, copied)
        assert _key(BiorbdModel(MODEL), options) == _key(BiorbdModel(str(copied)), options)

        text = original.read_text()
        assert "mass 1" in text, "the mass this test edits is no longer in the model"
        edited.write_text(text.replace("mass 1", "mass 2", 1))
        assert _key(BiorbdModel(MODEL), options) != _key(BiorbdModel(str(edited)), options)
    finally:
        copied.unlink(missing_ok=True)
        edited.unlink(missing_ok=True)


def test_key_changes_when_the_model_is_mutated_after_construction():
    """set_gravity does not invalidate _cached_functions, so the key has to notice it."""
    options = CodegenOptions()
    model = BiorbdModel(MODEL)
    before = _key(model, options)
    model.set_gravity(np.array([0, 0, -1.62]))
    assert _key(model, options) != before


def test_key_changes_with_compiler_flags():
    """A library built with different flags must never be served to a different build."""
    base = CodegenOptions()
    assert _key(BiorbdModel(MODEL), base) != _key(BiorbdModel(MODEL), CodegenOptions(opt_level=3))
    assert _key(BiorbdModel(MODEL), base) != _key(BiorbdModel(MODEL), CodegenOptions(extra_flags=("-march=native",)))


def test_key_changes_with_cache_tag():
    base = CodegenOptions()
    assert _key(BiorbdModel(MODEL), base) != _key(BiorbdModel(MODEL), CodegenOptions(cache_tag="experiment"))


def test_key_changes_with_the_model_class():
    """TorqueBiorbdModel and a bare BiorbdModel must not share compiled functions."""
    from bioptim import TorqueBiorbdModel

    options = CodegenOptions()
    assert _key(BiorbdModel(MODEL), options) != _key(TorqueBiorbdModel(MODEL), options)


def test_symbols_differ_for_different_kwargs():
    """
    Regression test for the bug that made the first attempt at this unusable.

    forward_dynamics(with_contact=True) and forward_dynamics(with_contact=False) have
    identical CasADi signatures. A cache keyed on the signature silently hands back the
    wrong one; the symbol has to be derived from the decorator's whole key.
    """
    fast_key = "deadbeef"
    with_contact = ("forward_dynamics", (), frozenset({("with_contact", True)}))
    without_contact = ("forward_dynamics", (), frozenset({("with_contact", False)}))
    no_kwargs = ("forward_dynamics", (), frozenset())

    symbols = {symbol_for(k, fast_key) for k in (with_contact, without_contact, no_kwargs)}
    assert len(symbols) == 3
    assert all(s.startswith("bioptim_forward_dynamics_") for s in symbols)


def test_symbols_differ_across_models():
    """The same method on two different models must not collide in one cache directory."""
    key = ("forward_dynamics", (), frozenset())
    assert symbol_for(key, "aaaa") != symbol_for(key, "bbbb")


def test_symbol_is_a_valid_c_identifier():
    symbol = symbol_for(("rotation_matrix_to_euler_angles", ("xyz",), frozenset()), "abc")
    assert symbol.isidentifier()


def test_codegen_off_by_default_leaves_the_model_untouched():
    model = BiorbdModel(MODEL)
    assert model._codegen is None
    assert model.forward_dynamics().class_name() == "MXFunction"


def test_codegen_disabled_explicitly():
    model = BiorbdModel(MODEL, codegen=False)
    assert model._codegen is None


def test_load_only_never_compiles(cache_dir):
    """load-only must degrade to plain Functions rather than invoking a compiler."""
    model = BiorbdModel(MODEL, codegen=CodegenOptions(cache_dir=cache_dir, mode="load-only"))
    assert model.forward_dynamics().class_name() == "MXFunction"
    model.codegen_flush()
    assert model._codegen.library_path is None


class TestCompiledRoundTrip:
    """These actually invoke a compiler, so they are slower than the rest of the file."""

    @staticmethod
    def _compiled_model(cache_dir):
        options = CodegenOptions(cache_dir=cache_dir, opt_level=0)
        cold = BiorbdModel(MODEL, codegen=options)
        cold.forward_dynamics()
        cold.markers()
        cold.codegen_flush()
        return BiorbdModel(MODEL, codegen=options)

    def test_warm_model_loads_externals_without_building_a_graph(self, cache_dir):
        warm = self._compiled_model(cache_dir)
        assert warm.forward_dynamics().class_name() == "External"
        assert warm.markers().class_name() == "External"

    def test_compiled_function_matches_the_original(self, cache_dir):
        warm = self._compiled_model(cache_dir)
        reference = BiorbdModel(MODEL)

        q = np.array([0.3, -0.7])
        qdot = np.array([0.1, 0.2])
        tau = np.array([1.0, -2.0])
        empty = np.zeros(0)

        expected = np.array(reference.forward_dynamics()(q, qdot, tau, empty, empty)).ravel()
        actual = np.array(warm.forward_dynamics()(q, qdot, tau, empty, empty)).ravel()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_compiled_function_keeps_its_named_signature(self, cache_dir):
        warm = self._compiled_model(cache_dir)
        compiled = warm.forward_dynamics()
        reference = BiorbdModel(MODEL).forward_dynamics()

        assert compiled.n_in() == reference.n_in()
        assert compiled.n_out() == reference.n_out()
        assert [compiled.name_in(i) for i in range(compiled.n_in())] == ["q", "qdot", "tau", "external_forces", "parameters"]

    def test_compiled_function_is_differentiable(self, cache_dir):
        """
        The whole point of emitting jac_/adj1_/fwd1_adj1_: without them the external has no
        derivatives at all, and without jac_ specifically its Jacobian is assumed dense.
        """
        warm = self._compiled_model(cache_dir)
        reference = BiorbdModel(MODEL)

        q, qdot, tau = MX.sym("q", 2), MX.sym("qdot", 2), MX.sym("tau", 2)
        empty = MX(0, 1)
        inputs = vertcat(q, qdot, tau)

        compiled_out = warm.forward_dynamics()(q, qdot, tau, empty, empty)
        reference_out = reference.forward_dynamics()(q, qdot, tau, empty, empty)

        compiled_jac = jacobian(compiled_out, inputs)
        reference_jac = jacobian(reference_out, inputs)

        # The compiled Jacobian must be at least as sparse as the reference: SX expansion
        # finds structural zeros the MX graph does not. A *denser* one means jac_ is missing.
        assert compiled_jac.nnz() <= reference_jac.nnz()

        point = (np.array([0.3, -0.7]), np.array([0.1, 0.2]), np.array([1.0, -2.0]))
        actual = np.array(Function("a", [q, qdot, tau], [compiled_jac])(*point))
        expected = np.array(Function("e", [q, qdot, tau], [reference_jac])(*point))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    def test_compiled_function_has_a_second_derivative(self, cache_dir):
        """IPOPT defaults to hessian_approximation='exact', so this must work."""
        from casadi import hessian, sum1

        warm = self._compiled_model(cache_dir)
        q, qdot, tau = MX.sym("q", 2), MX.sym("qdot", 2), MX.sym("tau", 2)
        out = warm.forward_dynamics()(q, qdot, tau, MX(0, 1), MX(0, 1))
        h, _ = hessian(sum1(out), vertcat(q, qdot, tau))
        assert h.nnz() > 0

    def test_second_model_reuses_the_library(self, cache_dir):
        """A warm model must not record anything: nothing left to compile means no graph built."""
        warm = self._compiled_model(cache_dir)
        warm.forward_dynamics()
        warm.markers()
        assert not warm._codegen.has_pending

    def test_cold_run_keeps_the_users_graph_settings(self, cache_dir):
        """
        A cold run must not be degraded.

        Forcing MX before anything is compiled gives the worst of both worlds: the model
        calls become interpreted call nodes rather than inlined expressions, which measures
        roughly 170x slower per constraint Jacobian on a whole-body model. So the override
        only applies once there is a library to override *for*.
        """
        ocp = _pendulum_ocp(CodegenOptions(cache_dir=cache_dir, opt_level=0), use_sx=True, expand=True)
        assert ocp.codegen_active is False
        assert ocp.cx.__name__ == "SX"
        assert ocp.nlp[0].dynamics_type.expand_dynamics is True

    def test_warm_run_switches_to_mx(self, cache_dir):
        options = CodegenOptions(cache_dir=cache_dir, opt_level=0)
        _pendulum_ocp(options, use_sx=True, expand=True)  # cold: builds and compiles
        ocp = _pendulum_ocp(options, use_sx=True, expand=True)  # warm

        assert ocp.codegen_active is True
        assert ocp.cx.__name__ == "MX"
        assert ocp.nlp[0].dynamics_type.expand_dynamics is False
        assert ocp.nlp[0].model.forward_dynamics(with_contact=False).class_name() == "External"

    def test_warm_run_reaches_the_same_optimum(self, cache_dir):
        from bioptim import Solver

        def solve(ocp):
            solver = Solver.IPOPT(show_online_optim=False)
            solver.set_print_level(0)
            return float(ocp.solve(solver).cost)

        options = CodegenOptions(cache_dir=cache_dir, opt_level=0)
        expected = solve(_pendulum_ocp(None, use_sx=True, expand=True))
        _pendulum_ocp(options, use_sx=True, expand=True)  # cold
        actual = solve(_pendulum_ocp(options, use_sx=True, expand=True))  # warm

        assert actual == pytest.approx(expected, rel=1e-8)

    def test_acados_refuses_a_compiled_model(self, cache_dir, monkeypatch):
        """
        ACADOS generates its own C from the SX graph, so it cannot resolve an external's
        symbols. Failing loudly at construction beats failing at link time.

        acados_template is stubbed so this runs whether or not acados is installed; the
        guard is the first statement of AcadosInterface.__init__, so nothing real is needed.
        """
        import sys
        import types

        stub = types.ModuleType("acados_template")
        for name in ("AcadosModel", "AcadosOcp", "AcadosOcpSolver"):
            setattr(stub, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, "acados_template", stub)

        from bioptim.interfaces.acados_interface import AcadosInterface

        options = CodegenOptions(cache_dir=cache_dir, opt_level=0)
        _pendulum_ocp(options, use_sx=True, expand=True)  # cold
        ocp = _pendulum_ocp(options, use_sx=True, expand=True)  # warm
        assert ocp.codegen_active is True

        with pytest.raises(RuntimeError, match="not compatible with compiled model functions"):
            AcadosInterface(ocp)

    def test_subprocess_mode_produces_the_same_library(self, cache_dir):
        """
        The default build runs in a child process so the parent never grows an allocator
        arena it cannot give back. The functions travel by CasADi serialization, not by
        rebuilding the model, so the child needs no backend at all.
        """
        options = CodegenOptions(cache_dir=cache_dir, opt_level=0, mode="subprocess")
        cold = BiorbdModel(MODEL, codegen=options)
        cold.forward_dynamics()
        cold.codegen_flush()

        warm = BiorbdModel(MODEL, codegen=options)
        compiled = warm.forward_dynamics()
        assert compiled.class_name() == "External"

        reference = BiorbdModel(MODEL).forward_dynamics()
        point = (np.array([0.3, -0.7]), np.array([0.1, 0.2]), np.array([1.0, -2.0]), np.zeros(0), np.zeros(0))
        np.testing.assert_allclose(
            np.array(compiled(*point)).ravel(), np.array(reference(*point)).ravel(), rtol=1e-12, atol=1e-12
        )

    def test_subprocess_build_works_from_any_working_directory(self, cache_dir, tmp_path, monkeypatch):
        """
        A child inherits environment variables but not sys.path, so a bioptim found through a
        path manipulation -- a checkout, an editable install, a test harness -- would not be
        importable in the build subprocess. The failure is silent: it warns and falls back to
        uncompiled functions, so only the missing speed-up would show it.
        """
        monkeypatch.chdir(tmp_path)
        options = CodegenOptions(cache_dir=cache_dir, opt_level=0, mode="subprocess")

        import warnings

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            cold = BiorbdModel(MODEL, codegen=options)
            cold.forward_dynamics()
            cold.codegen_flush()
        assert not [w for w in recorded if "subprocess failed" in str(w.message)]

        warm = BiorbdModel(MODEL, codegen=options)
        assert warm.forward_dynamics().class_name() == "External"

    def test_key_is_resolved_lazily(self, cache_dir):
        """
        A model is not finished when __init__ returns -- gravity can change, and a holonomic
        model does not know its constraints yet. So the key must be computed on first use.
        """
        options = CodegenOptions(cache_dir=cache_dir, opt_level=0)
        model = BiorbdModel(MODEL, codegen=options)
        assert model._codegen._fast_key is None  # nothing computed yet

        model.set_gravity(np.array([0, 0, -1.62]))
        moon = model._codegen.fast_key

        earth_model = BiorbdModel(MODEL, codegen=options)
        assert earth_model._codegen.fast_key != moon

    def test_rebuild_discards_the_existing_entry(self, cache_dir):
        self._compiled_model(cache_dir)
        options = CodegenOptions(cache_dir=cache_dir, opt_level=0, rebuild=True)
        model = BiorbdModel(MODEL, codegen=options)
        assert model._codegen.library_path is None
        assert model.forward_dynamics().class_name() == "MXFunction"
