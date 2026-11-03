import importlib.util
import math
import unittest
from unittest.mock import patch
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v6_high_fidelity_simulator",HERE/"v6_high_fidelity_simulator.py")
hf=importlib.util.module_from_spec(spec);spec.loader.exec_module(hf)


class HighFidelityTests(unittest.TestCase):
    def test_verified_statevector(self):
        state,evidence=hf.verified_statevector([float(i%7) for i in range(256)],math.pi/7,math.pi/11)
        self.assertEqual(len(state),256)
        self.assertGreaterEqual(evidence["fidelity"],0.99999)
        self.assertLess(abs(evidence["state_norm"]-1),1e-9)

    def test_guards(self):
        with self.assertRaises(ValueError): hf.verified_statevector([1,2,3],.1,.2)
        with self.assertRaises(ValueError): hf.verified_statevector([1,2,3,4],.1,.2,minimum_fidelity=0)
        with self.assertRaises(ValueError): hf._fidelity([0j],[0j])

    def test_verification_failures(self):
        with patch.object(hf,"_statevector",return_value=[2+0j,0j,0j,0j]):
            with self.assertRaises(RuntimeError): hf.verified_statevector([1,2,3,4],.1,.2)
        with patch.object(hf,"_fidelity",return_value=(0.5,1.0)):
            with self.assertRaises(RuntimeError): hf.verified_statevector([1,2,3,4],.1,.2)


if __name__=="__main__": unittest.main()
