import unittest
import numpy as np
from context_prediction.e4_two_image_precheck import in_box, match_same_class, missing_vote
from context_prediction.e4_window_intersection import intersecting


class WindowComparisonTests(unittest.TestCase):
    def test_crossing_component_uses_full_mask_then_clips(self):
        mask=np.zeros((256,256),bool); mask[90:160,40:80]=True
        component=dict(id='3:1',class_id=3,mask=mask,center=[60/256,125/256],area=mask.mean(),color=[0,0,0],appearance=[1.,0.])
        box=(0.,.5,.5,1.); clip=np.zeros_like(mask);clip[128:,:128]=True
        self.assertEqual(in_box([component],box),[])
        found=intersecting([component],clip)
        self.assertEqual(len(found),1)
        np.testing.assert_array_equal(found[0]['mask'],mask)
        pairs,missing=match_same_class(found,found)
        self.assertEqual(pairs,{0:0});self.assertEqual(missing,[])
        _,difference=missing_vote(found,found,pairs,np.ones_like(mask))
        self.assertFalse((difference&clip).any())


if __name__=='__main__':unittest.main()
