"""Exercise legacy LSTM/GRU orchestration with an inert torch test double.

This does not perform learning, enable ABE in V6, or assert prediction accuracy.
Only verifies shapes, training-loop control flow, metadata, and output contracts.
"""
import contextlib
import sys
import types
import unittest
from unittest.mock import patch

import abe_train_lstm_gru as legacy


class Tensor:
    def __init__(self, data): self.data = data
    def __getitem__(self, index):
        if isinstance(index, tuple):
            return Tensor([[0.0] * 4])
        return Tensor(self.data[index])
    def tolist(self): return self.data


class Module:
    def __call__(self, *args): return self.forward(*args)
    def parameters(self): return []
    def train(self): pass
    def eval(self): pass


class Recurrent(Module):
    def __init__(self, *args, **kwargs): pass
    def forward(self, seq): return Tensor([[[0.0]*4]*20]), None


class Linear(Module):
    def __init__(self, unused, outputs): self.outputs=outputs
    def forward(self, seq): return Tensor([[0.0]*self.outputs])


class Loss:
    def __add__(self, other): return self
    def backward(self): pass


class Optimizer:
    def __init__(self, *args, **kwargs): pass
    def zero_grad(self): pass
    def step(self): pass


def fake_torch():
    torch=types.ModuleType("torch")
    torch.float32="float32"; torch.long="long"
    torch.manual_seed=lambda seed:None
    torch.set_num_threads=lambda threads:None
    torch.tensor=lambda values, dtype=None:Tensor(values)
    torch.sigmoid=lambda values:Tensor([0.5]*len(values.data))
    torch.softmax=lambda values, dim=None:Tensor([1.0/len(values.data)]*len(values.data))
    torch.no_grad=contextlib.nullcontext
    nn=types.ModuleType("torch.nn")
    nn.Module=Module; nn.LSTM=Recurrent; nn.GRU=Recurrent; nn.Linear=Linear
    nn.functional=types.SimpleNamespace(
        binary_cross_entropy_with_logits=lambda *args:Loss(),
        cross_entropy=lambda *args:Loss())
    torch.nn=nn
    torch.optim=types.SimpleNamespace(Adam=Optimizer)
    return torch, nn


class LegacyTrainingWithTestDouble(unittest.TestCase):
    def test_training_control_flow_and_output_contract(self):
        draw_list=[{"draw_id":str(i), "first":[1,2,3,4,5,6], "second":1}
                   for i in range(25)]
        torch,nn=fake_torch()
        with patch.dict(sys.modules,{"torch":torch,"torch.nn":nn}):
            output=legacy.run(draw_list, epochs=2, seed=7, hidden=4)
        self.assertEqual(output["through_draw"],"24")
        self.assertEqual(output["training_samples"],5)
        self.assertEqual(output["features_per_sample"],20)
        self.assertEqual(output["epochs"],2)
        self.assertEqual(set(output["models"]),{"lstm","gru"})
        for name in ("lstm","gru"):
            self.assertEqual(len(output["models"][name]["first"]),38)
            self.assertEqual(len(output["models"][name]["second"]),8)
        self.assertFalse(output["five_model_fusion_executed"])


if __name__=="__main__":
    unittest.main()
