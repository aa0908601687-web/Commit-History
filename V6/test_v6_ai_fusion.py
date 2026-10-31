import unittest
from v6_ai_fusion import fusion_scores,predict_fusion
class FusionTests(unittest.TestCase):
    def make_draws(self):
        return [{"draw_id":str(i),"first":[((i+6*j)%38)+1 for j in range(6)],"second":(i%8)+1} for i in range(12)]
    def test_fusion(self):
        data=self.make_draws()
        prediction=predict_fusion(data)
        self.assertEqual(len(prediction["first"]),6)
        self.assertEqual(len(set(prediction["first"])),6)
        self.assertTrue(1<=prediction["second"]<=8)
        self.assertFalse(prediction["trained_ai_integrated"])
        self.assertEqual(prediction,predict_fusion(data))
    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            fusion_scores(self.make_draws(),"first",(0.5,0.5,0.5))
if __name__=="__main__":unittest.main()
