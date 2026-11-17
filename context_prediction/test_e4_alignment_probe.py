import unittest
import numpy as np
import cv2
from context_prediction.e4_normal_alignment_probe import connector_transform


class AlignmentChecks(unittest.TestCase):
    def test_identity_and_translation(self):
        def anchors(x,y):return [{'center':np.array([x,y])},{'center':np.array([x+80,y+10])}]
        src=anchors(40.5,80.5)
        identity,scale,angle=connector_transform(src,src)
        np.testing.assert_allclose(identity,np.array([[1,0,0],[0,1,0]]),atol=1e-10)
        target=anchors(47.5,77.5)
        matrix,scale,angle=connector_transform(src,target)
        np.testing.assert_allclose(matrix,np.array([[1,0,7],[0,1,-3]]),atol=1e-10)
        mask=np.zeros((256,256),np.uint8);mask[80,40]=1
        moved=cv2.warpAffine(mask,matrix,(256,256),flags=cv2.INTER_NEAREST)
        self.assertEqual(moved[77,47],1)
        self.assertEqual(int(moved.sum()),1)


if __name__=='__main__':unittest.main()
