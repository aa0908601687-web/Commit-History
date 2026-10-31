import unittest
from v6_neural_train import train
class NeuralTests(unittest.TestCase):
    def draws(self):
        return [{"draw_id":str(i),"first":[((i+j*6)%38)+1 for j in range(6)],"second":(i%8)+1} for i in range(18)]
    def test_training_and_qaoa(self):
        result=train(self.draws(),epochs=2,hidden=5)
        self.assertEqual(result["training_samples"],8)
        self.assertEqual(len(result["first"]),6)
        self.assertEqual(len(set(result["first"])),6)
        self.assertTrue(1<=result["second"]<=8)
        self.assertTrue(result["trained_ai_integrated"])
        self.assertFalse(result["predictive_advantage_verified"])
        self.assertEqual(result,train(self.draws(),epochs=2,hidden=5))
    def test_small_data(self):
        with self.assertRaises(ValueError):train(self.draws()[:10])
if __name__=="__main__":unittest.main()
