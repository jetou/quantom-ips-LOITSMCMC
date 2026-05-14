from dataclasses import dataclass, field, asdict, make_dataclass
import dataclasses
import importlib
import copy
import pprint
from hydra.core.config_store import ConfigStore
from typing import Optional
import yaml
import inspect
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)

registry = {}


@dataclass
class ModuleSpec:
    id: str
    entry_point: str
    group: str
    kwargs: dict = field(default_factory=dict)


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def register(
    id: str,
    entry_point: str,
    group: Optional[str] = None,
    config: Optional[str] = None,
    kwargs: Optional[dict] = None,
    defaults: Optional[list] = None,
):
    # logger.info(f"Registering {id}: <entry_point>={entry_point}, <group>={group}")
    if config is not None:
        if isinstance(config, str):
            config_kwargs = OmegaConf.load(config)
        elif dataclasses.is_dataclass(config):
            config_kwargs = OmegaConf.create(config)
        else:
            raise ValueError("config must be a path to a yaml file or a dataclass")
    else:
        # If no config, try to load the entry point and create a dataclass with it
        try:
            cls = load(entry_point)
            config = get_dataclass_from_class(cls)
        except Exception as e:
            print(f"Failed to register {group}/{id}: {str(e)}")
            return

        config_kwargs = OmegaConf.create(config)

    if "id" in config_kwargs:
        raise ValueError("Key 'id' cannot be used in the yaml_file")
    if "entry_point" in config_kwargs:
        raise ValueError("Key 'entry_point' cannot be used in the yaml_file")

    mod_kwargs = {
        "id": id,
        "entry_point": entry_point,
    }

    mod_kwargs = OmegaConf.merge(mod_kwargs, config_kwargs)
    if kwargs is not None:
        mod_kwargs = OmegaConf.merge(mod_kwargs, kwargs)
    if defaults is not None:
        OmegaConf.set_struct(mod_kwargs, False)
        mod_kwargs["defaults"] = defaults
        OmegaConf.set_struct(mod_kwargs, True)

    registry[id] = ModuleSpec(id, entry_point, group, mod_kwargs)
    cs = ConfigStore.instance()
    cs.store(name=id, group=group, node=mod_kwargs)


def make(kwargs: Optional[dict] = None):
    # Check if kwargs is a dictionary, if not, return it as is
    if kwargs is not None and not isinstance(kwargs, (dict, DictConfig)):
        print(f"WARNING: kwargs type ({type(kwargs)}) is not a dict, returning as is.")
        return kwargs

    # Check if "id" is registered
    spec = registry.get(kwargs["id"])
    if spec is None:
        logger.warning(f"Module with id: {kwargs['id']} not found in registry.")

    # Attempt to load module (regardless of whether spec was found)
    mod = load(kwargs["entry_point"])

    mod_kwargs = {}
    mod_kwargs.update(kwargs)

    # print("REGISTRY make():", id, mod_kwargs)

    mod_kwargs.pop("id")
    mod_kwargs.pop("entry_point")

    return mod(**mod_kwargs)


def list_modules():
    modules = list(registry.keys())
    spec_list = [registry.get(m) for m in modules]
    groups = set([s.group for s in spec_list])
    groups = sorted(groups)
    for g in groups:
        print(f"Group: {g}")
        for s in spec_list:
            if s.group == g:
                print(f"- {s.id}")


def get_dataclass_from_class(cls):
    # Get the signature of the class's __init__ method
    signature = inspect.signature(cls.__init__)
    cls_name = cls.__name__

    fields_list = []
    for param in signature.parameters.values():
        if param.name == "self":
            continue  # Skip 'self' parameter

        if (
            param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or param.kind == inspect.Parameter.KEYWORD_ONLY
        ):
            if param.default is inspect.Parameter.empty:
                # Required field
                fields_list.append((param.name, param.annotation))
            else:
                # Field with default value
                # Check if annotation is list or dict to use default_factory
                if param.annotation == list:
                    fields_list.append(
                        (
                            param.name,
                            param.annotation,
                            field(default_factory=lambda x=param.default: list(x)),
                        )
                    )
                elif param.annotation == dict:
                    fields_list.append(
                        (
                            param.name,
                            param.annotation,
                            field(default_factory=lambda x=param.default: dict(x)),
                        )
                    )
                else:
                    fields_list.append(
                        (param.name, param.annotation, field(default=param.default))
                    )

    return make_dataclass(cls_name + "Cfg", fields_list)
