from bioptim.misc.codegen_cache import FunctionCodegenCache
from bioptim import BiorbdModel
from bioptim.examples.utils import ExampleUtils

model_path = ExampleUtils.folder + "/models/pendulum.bioMod"
# Create a shared cache
cache = FunctionCodegenCache(cache_dir=".bioptim_cache", verbose=True)
# First run: generates and compiles C code
model = BiorbdModel(model_path, codegen_cache=cache)
model.forward_dynamics()
# Second run: loads from cached .so files
model2 = BiorbdModel(model_path, codegen_cache=cache)
model2.forward_dynamics()