import unittest
import numpy as np
from context_prediction.e4_two_image_precheck import match_same_class,missing_vote


def comp(name,mask,cls=2):
    return dict(id=name,class_id=cls,mask=mask,center=[.5,.5],area=.1,color=[0,0,0],appearance=[1.,0.])


class E4Tests(unittest.TestCase):
    def test_matched_missing_and_clipping(self):
        mask=np.zeros((32,32),bool); mask[8:24,8:24]=True
        clip=np.ones((32,32),bool); a=comp('a',mask); matched,missing=match_same_class([a],[a])
        self.assertEqual(matched,{0:0}); self.assertFalse(missing)
        _,same=missing_vote([a],[a],matched,clip); self.assertFalse(same.any())
        _,absent=missing_vote([a],[],{},clip); np.testing.assert_array_equal(absent,mask)
        partial=mask.copy(); partial[:,16:]=False
        _,difference=missing_vote([a],[comp('partial',partial)],{0:0},clip)
        self.assertTrue(difference.any()); self.assertFalse(difference[:, :16].any())
        clip[:,20:]=False; _,clipped=missing_vote([a],[],{},clip); self.assertFalse(clipped[:,20:].any())

    def test_different_classes_do_not_cancel(self):
        mask=np.ones((12,12),bool); a=comp('a',mask,0); b=comp('b',mask,1)
        matched,missing=match_same_class([a],[b]); self.assertEqual(matched,{}); self.assertEqual(missing,[0])


if __name__=='__main__': unittest.main()
