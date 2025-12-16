"""CasADi Function code generation and caching utilities.

This module provides utilities for generating C code from CasADi Functions,
compiling them to shared libraries, and caching them for reuse across OCP
instantiations.

Example
-------
>>> from bioptim.misc.codegen_cache import FunctionCodegenCache
>>> cache = FunctionCodegenCache(cache_dir=".bioptim_cache")
>>> model = BiorbdModel("model.bioMod", codegen_cache=cache)
"""

from __future__ import annotations
import hashlib
import os
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from casadi import Function, CodeGenerator, external

if TYPE_CHECKING:
    from ..models.biorbd.biorbd_model import BiorbdModel


class FunctionCodegenCache:
    """
    Cache for compiled CasADi Functions.

    Generates C code from Functions, compiles to shared libraries,
    and reuses them across OCP instantiations. This can significantly
    reduce memory usage and improve evaluation speed for repeated solves.

    Parameters
    ----------
    cache_dir : str
        Directory to store cached .c and .so files.
        Default is ".bioptim_cache" in the current working directory.
    verbose : bool
        If True, print information about cache hits/misses and compilation.

    Attributes
    ----------
    cache_dir : Path
        The cache directory path.

    Example
    -------
    >>> cache = FunctionCodegenCache(cache_dir=".bioptim_cache")
    >>> func = model.forward_dynamics()  # Creates MX Function
    >>> compiled_func = cache.get_or_compile(func, model)
    """

    def __init__(self, cache_dir: str = ".bioptim_cache", verbose: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._loaded_functions: dict[str, Function] = {}
        self._enabled = True  # Can be disabled for SX mode (external functions don't support eval_sx)

    def get_cache_key(self, func: Function, model_path: str) -> str:
        """
        Generate unique cache key based on function and model.

        The key is a hash of:
        - Model file path
        - Function name
        - Number of inputs/outputs
        - Size of each input

        Parameters
        ----------
        func : Function
            The CasADi Function to generate a key for.
        model_path : str
            Path to the model file.

        Returns
        -------
        str
            A 12-character hex hash string.
        """
        # Build signature string
        signature = f"{model_path}:{func.name()}:{func.n_in()}:{func.n_out()}"
        for i in range(func.n_in()):
            size = func.size_in(i)
            signature += f":{size[0]}x{size[1]}"
        for i in range(func.n_out()):
            size = func.size_out(i)
            signature += f":{size[0]}x{size[1]}"

        return hashlib.md5(signature.encode()).hexdigest()[:12]

    @property
    def enabled(self) -> bool:
        """Whether codegen is enabled. Disable for SX mode (external functions don't support eval_sx)."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def get_or_compile(self, func: Function, model: "BiorbdModel") -> Function:
        """
        Get compiled function from cache or compile it.

        If a compiled version exists in the cache, it is loaded and returned.
        Otherwise, the function is compiled to C, then to a shared library,
        and the compiled version is returned.

        Parameters
        ----------
        func : Function
            The CasADi Function to compile/retrieve.
        model : BiorbdModel
            The model associated with this function (used for cache key).

        Returns
        -------
        Function
            The compiled CasADi Function loaded from shared library,
            or the original function if compilation fails or is disabled.
        """
        # Skip codegen if disabled (e.g., for SX mode which doesn't support external functions)
        if not self._enabled:
            return func
        
        func_name = func.name()
        cache_key = self.get_cache_key(func, model.path)
        so_path = self.cache_dir / f"{func_name}_{cache_key}.so"

        # Check in-memory cache first
        cache_id = str(so_path)
        if cache_id in self._loaded_functions:
            if self.verbose:
                print(f"[codegen_cache] Memory hit: {func_name}")
            return self._loaded_functions[cache_id]

        # Check if .so exists on disk
        if so_path.exists():
            if self.verbose:
                print(f"[codegen_cache] Disk hit: {so_path}")
            try:
                compiled = self._load_from_so(so_path, func_name)
                self._loaded_functions[cache_id] = compiled
                return compiled
            except Exception as e:
                warnings.warn(f"Failed to load cached function {so_path}: {e}")
                # Fall through to recompile

        # Generate and compile
        if self.verbose:
            print(f"[codegen_cache] Compiling: {func_name} -> {so_path}")
        try:
            compiled = self._compile_function(func, cache_key)
            self._loaded_functions[cache_id] = compiled
            return compiled
        except Exception as e:
            warnings.warn(
                f"Failed to compile function {func_name}: {e}. "
                "Falling back to interpreted function."
            )
            return func

    def _compile_function(self, func: Function, cache_key: str) -> Function:
        """
        Generate C code and compile to shared library.

        This method compiles the function and its derivative chain:
        1. Main function (for evaluation)
        2. adj1_<name> (reverse derivative for gradients)
        3. fwd1_adj1_<name> (forward derivative of reverse for Hessians)

        CasADi automatically looks for these naming conventions in external functions.

        Parameters
        ----------
        func : Function
            The CasADi Function to compile.
        cache_key : str
            The unique cache key for this function.

        Returns
        -------
        Function
            The compiled CasADi Function.
        """
        func_name = func.name()
        file_base = f"{func_name}_{cache_key}"
        c_file = self.cache_dir / f"{file_base}.c"
        so_file = self.cache_dir / f"{file_base}.so"

        # Expand MX to SX for proper code generation
        # (MX functions fail to load with external())
        try:
            func_expanded = func.expand()
        except RuntimeError:
            raise RuntimeError(
                f"Cannot expand function {func_name} for code generation. "
                "This function may contain operations that cannot be converted to SX."
            )

        # Generate C code for main function AND its derivative chain
        cg = CodeGenerator(
            file_base,
            {"mex": False}
        )
        
        # Add main function
        cg.add(func_expanded)
        if self.verbose:
            print(f"[codegen_cache] Added function: {func_name}")

        # Add reverse derivative (adj1_<name>) for gradient computation
        try:
            adj1_func = func_expanded.reverse(1)
            cg.add(adj1_func)
            if self.verbose:
                print(f"[codegen_cache] Added reverse derivative: {adj1_func.name()}")
            
            # Add forward derivative of reverse (fwd1_adj1_<name>) for Hessian computation
            try:
                fwd1_adj1_func = adj1_func.forward(1)
                cg.add(fwd1_adj1_func)
                if self.verbose:
                    print(f"[codegen_cache] Added fwd of reverse: {fwd1_adj1_func.name()}")
            except Exception as e:
                if self.verbose:
                    print(f"[codegen_cache] Could not generate fwd1_adj1_{func_name}: {e}")
        except Exception as e:
            if self.verbose:
                print(f"[codegen_cache] Could not generate adj1_{func_name}: {e}")

        cg.generate(str(self.cache_dir) + os.sep)

        # Compile to shared library
        self._compile_c_to_so(c_file, so_file)

        return self._load_from_so(so_file, func_name)

    def _compile_c_to_so(self, c_file: Path, so_file: Path) -> None:
        """
        Compile C source to shared library using gcc.

        Parameters
        ----------
        c_file : Path
            Path to the C source file.
        so_file : Path
            Path for the output shared library.

        Raises
        ------
        RuntimeError
            If compilation fails.
        """
        cmd = [
            "gcc",
            "-fPIC",
            "-shared",
            "-O3",
            "-march=native",
            str(c_file),
            "-o",
            str(so_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")

    def _load_from_so(self, so_path: Path, func_name: str) -> Function:
        """
        Load compiled function from shared library.

        Parameters
        ----------
        so_path : Path
            Path to the shared library.
        func_name : str
            Name of the function to load.

        Returns
        -------
        Function
            The loaded CasADi Function.
        """
        # Derivatives are compiled into the .so file (adj1_, fwd1_adj1_)
        # No need for enable_fd since we provide full AD support
        return external(func_name, str(so_path))

    def clear(self) -> None:
        """Clear all cached functions from memory (disk cache remains)."""
        self._loaded_functions.clear()

    def clear_disk(self) -> None:
        """Remove all cached files from disk."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_functions.clear()
