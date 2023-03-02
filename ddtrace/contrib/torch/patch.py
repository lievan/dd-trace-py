from ddtrace import config

from .. import trace_utils

import torch

from ddtrace import config
from ddtrace.pin import Pin
from ddtrace.vendor.wrapt import wrap_function_wrapper as _w
from ddtrace.internal.constants import COMPONENT

from ...internal.utils.wrappers import unwrap as _u


config._add(
    "torch",
    {
        "_default_service": "torch",
    },
)


def patch():
    """Enable tracing for PyTorch nn.Module"""
    if getattr(torch, "__datadog_patch", False):
        return
    setattr(torch, "__datadog_patch", True)
    _w("torch", "nn.Module.__init__", _wrap_nn_init)
    Pin().onto(torch.nn.Module)


def unpatch():
    """Disable tracing for PyTorch nn.Module"""
    if getattr(torch, "__datadog_patch", False):
        setattr(torch, "__datadog_patch", False)
        _u(torch.nn.Module, "__init__")


def _wrap_nn_init(func, instance, args, kwargs):
    """
    Wrapped function for __init__ for nn.Module
    """
    # dont need to inspect args
    pin = Pin.get_from(instance)
    with pin.tracer.trace("model.init", service=trace_utils.ext_service(pin, config.torch)) as span:
        span.set_tag_str(COMPONENT, config.torch.integration_name)
        print(span)
    func(*args, **kwargs)
