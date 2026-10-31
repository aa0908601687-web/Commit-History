import unittest
from v6_learned_stacking import crossfit_predictions,learn_weights,predict
class LearnedStackingTests(unittest.TestCase):
    def draws(self):
        return [{"draw_id":str(i),"first":[((i+j*6)%38)+1 for j in range(6)],"second":i%8+1} for i in range(16)]
    def test_learned_meta_weights(self):
        data=self.draws()
        folds=crossfit_predictions(data,max_folds=4)
        self.assertEqual(len(folds),4)
        self.assertEqual(folds[0][2],"12")
        w=learn_weights(folds)
        self.assertEqual(len(w),3)
        self.assertAlmostEqual(sum(w),1.0,places=10)
        self.assertTrue(all(0<=v<=1 for v in w))
        result=predict(data)
        self.assertEqual(len(result["first"]),6)
        self.assertTrue(1<=result["second"]<=8)
        self.assertTrue(result["meta_training_only_on_past_draws"])
        self.assertFalse(result["predictive_advantage_verified"])
        self.assertEqual(result,predict(data))
    def test_small_input_rejected(self):
        with self.assertRaises(ValueError):crossfit_predictions(self.draws()[:14])
if __name__=="__main__":unittest.main()
