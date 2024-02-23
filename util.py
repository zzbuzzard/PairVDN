import jax
import equinox as eqx
from os.path import join, isfile


def save_model(root_path: str, model: eqx.Module):
    path = join(root_path, "model.eqx")
    print("Serialising model to", path)
    eqx.tree_serialise_leaves(path, model)


def load_model(root_path: str, model: eqx.Module) -> eqx.Module:
    path = join(root_path, "model.eqx")
    if isfile(path):
        print("Deserialising from", path)
        return eqx.tree_deserialise_leaves(path, model)
    else:
        print(path, "does not exist, not loading model")
        return model

