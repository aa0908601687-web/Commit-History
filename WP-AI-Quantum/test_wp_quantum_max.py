import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from wp_quantum_max import select
class TestQuantum(unittest.TestCase):
    def setUp(self):
        self.draws=[(str(i),tuple(sorted({(i*7+j*5)%38+1 for j in range(6)})),i%8+1) for i in range(20)]
    def test_deterministic(self):
        self.assertEqual(select(self.draws),select(self.draws))
    def test_valid_output(self):
        result=select(self.draws)
        self.assertEqual(len(set(result["first"])),6)
        self.assertTrue(all(1<=n<=38 for n in result["first"]))
        self.assertTrue(1<=result["second"]<=8)
        self.assertFalse(result["abe_used"])
    def test_short_history(self):
        with self.assertRaises(ValueError):select(self.draws[:19])
if __name__=="__main__":unittest.main()
