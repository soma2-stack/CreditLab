"""Non-training checks for the four-cell HARD_V2 benchmark."""
import importlib.util
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
SPEC=importlib.util.spec_from_file_location("koth",ROOT/"experiments"/"run_koth_four_cell.py")
koth=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(koth)

class FourCellChecks(unittest.TestCase):
    def test_four_unique_cells(self):
        cells=[(v,s) for v in koth.VARIANTS for s in koth.SEEDS]
        self.assertEqual(len(cells),4)
        self.assertEqual(len(set(cells)),4)
    def test_noise_strength_and_split_reproducibility(self):
        import torch
        for variant in koth.VARIANTS:
            x,y=koth.split(32,16,17,0,variant)
            x2,y2=koth.split(32,16,17,0,variant)
            self.assertTrue(torch.equal(x,x2))
            self.assertTrue(torch.equal(y,y2))
            self.assertEqual(tuple(x.shape),(32,18,3))
            self.assertFalse(torch.equal(x,koth.split(32,16,17,1,variant)[0]))
        self.assertFalse(torch.equal(koth.split(32,16,17,0,"original")[0],
                                         koth.split(32,16,17,0,"fixed")[0]))
    def test_no_training_by_default(self):
        import subprocess
        p=subprocess.run([sys.executable,str(ROOT/"experiments"/"run_koth_four_cell.py")],
                         capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('"training_started": false',p.stdout.lower())
    def test_bad_parameters_refused(self):
        import subprocess
        p=subprocess.run([sys.executable,str(ROOT/"experiments"/"run_koth_four_cell.py"),
                          "--updates","0"],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(p.returncode,0)

if __name__=="__main__":
    unittest.main()
