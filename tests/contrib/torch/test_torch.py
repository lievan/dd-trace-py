import ddtrace
import torch

from ddtrace.contrib.torch.patch import patch
from ddtrace.contrib.torch.patch import unpatch
from tests.utils import TracerTestCase
from ddtrace.pin import Pin


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h


class TestTorch(TracerTestCase):
    def setUp(self):
        super(TestTorch, self).setUp()
        patch()
        Pin.override(torch.nn.Module, tracer=self.tracer)

    def tearDown(self):
        super(TestTorch, self).tearDown()
        unpatch()

    def test_model_init(self):
        MyCell()
        spans = self.pop_spans()
        assert len(spans) == 1
        s = spans[0]
