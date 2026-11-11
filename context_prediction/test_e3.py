import unittest
import numpy as np
from context_prediction.e3_evaluate import error, indices, queries, distance


def example(center=(.1,.1), appearance=(1.,0.)):
    return dict(region_centers=np.array([center]),region_features=np.array([appearance]),
                region_lab=np.zeros((1,3)),region_area=np.array([.1]),
                tile_features=np.tile(np.array([[1.,0.]]),(16,1)),tile_lab=np.zeros((16,3)))


class E3Checks(unittest.TestCase):
    def test_matching_and_empty(self):
        a=example(); self.assertAlmostEqual(error(a,a,(0,0,.25,.25)),0.)
        b=example(center=(.8,.8)); self.assertAlmostEqual(error(a,b,(0,0,.25,.25)),.35)
        self.assertEqual(error(b,b,(0,0,.25,.25)),0.)
        self.assertEqual(indices(a,(0,0,.25,.25)).tolist(),[0])

    def test_windows_and_hidden_tile(self):
        q=queries(); self.assertEqual(len(q),20)
        self.assertEqual(len(q[-1][1]),4)
        a=example(); b=example(); b['tile_features'][0]=[-1,0]
        bank=[(type('P',(),{'name':'z'})(),a),(type('P',(),{'name':'a'})(),b)]
        self.assertEqual([p for _,p in distance(a,bank,list(range(1,16)))],['a','z'])
        self.assertEqual(distance(a,bank,list(range(16)))[0][1],'z')


if __name__=='__main__': unittest.main()
