import unittest
from v6_ensemble_v2 import predict,WEIGHTS
class EnsembleV2Tests(unittest.TestCase):
    def draws(self):
        return [{"draw_id":str(i),"first":[((i+j*6)%38)+1 for j in range(6)],"second":(i%8)+1} for i in range(13)]
    def test_weights_and_prediction(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()),1.0)
        self.assertNotIn("repeat",WEIGHTS)
        result=predict(self.draws())
        self.assertEqual(len(result["first"]),6)
        self.assertEqual(len(set(result["first"])),6)
        self.assertTrue(all(1<=v<=38 for v in result["first"]))
        self.assertTrue(1<=result["second"]<=8)
        self.assertEqual(result["trained_neural_models"],3)
        self.assertFalse(result["stacking_weights_learned"])
        self.assertFalse(result["predictive_advantage_verified"])
        self.assertEqual(result,predict(self.draws()))
if __name__=="__main__":unittest.main()
