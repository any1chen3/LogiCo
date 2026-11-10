import unittest
import numpy as np

from context_prediction.evidence import component_cost, fuse_with_evidence, match, rasterize_box, resize_mask
from context_prediction.predictor import retrieve


def component(name, appearance=(1., 0.), color=(0., 0., 0.), center=(.5, .5), area=.1):
    return dict(id=name, appearance=appearance, color=color, center=center, area=area)


class E1Tests(unittest.TestCase):
    def setUp(self):
        self.t1 = component('T1')
        self.t2 = component('T2', appearance=(-1., 0.), color=(100., 0., 0.))
        self.refs = [dict(id='C1', visible_key=[0., 1.], expected=[self.t1]),
                     dict(id='C2', visible_key=[1., 0.], expected=[self.t2])]

    def test_context_target_interventions(self):
        cases = [([0., 1.], self.t1, 'C1', True), ([0., 1.], self.t2, 'C1', False),
                 ([1., 0.], self.t2, 'C2', True)]
        for visible, target, ref_id, same in cases:
            ref = retrieve(visible, self.refs)[0]
            self.assertEqual(ref['id'], ref_id)
            self.assertEqual(ref['expected'][0]['id'], 'T1' if ref_id == 'C1' else 'T2')
            self.assertEqual(component_cost(ref['expected'][0], target) < .01, same)
        self.assertEqual(match([self.t1], [self.t2])['missing'], ['T1'])
        self.assertEqual(match([self.t1], [self.t2])['extra'], ['T2'])

    def test_missing_extra_empty_and_permutation(self):
        a, b = component('a'), component('b', center=(.2, .2))
        result = match([b, a], [a, b])
        self.assertEqual(result, match([a, b], [b, a]))
        self.assertEqual([(p[0], p[1]) for p in result['pairs']], [('a', 'a'), ('b', 'b')])
        self.assertEqual(match([a, b], [a])['missing'], ['b'])
        self.assertEqual(match([a], [a, b])['extra'], ['b'])
        self.assertEqual(match([], []), {'pairs': [], 'missing': [], 'extra': []})
        self.assertEqual(match([], [a])['extra'], ['a'])
        self.assertEqual(match([a], [])['missing'], ['a'])

    def test_duplicate_appearance_and_tie_stability(self):
        a, b = component('a'), component('b')
        x, y = component('x'), component('y')
        self.assertEqual(match([a, b], [x, y]), match([b, a], [y, x]))
        refs = [dict(id='z', visible_key=[0], expected=[]), dict(id='a', visible_key=[0], expected=[])]
        self.assertEqual(retrieve([0], refs)[0]['id'], 'a')

    def test_mask_geometry(self):
        for shape in [(12, 20), (20, 12)]:
            mask = rasterize_box((-2, 2, 5, 7), shape)
            self.assertEqual(int(mask.sum()), 25)
            self.assertTrue(mask[2:7, :5].all())
            self.assertEqual(mask[7, 4], 0)
            enlarged = resize_mask(mask, (shape[0] * 2, shape[1] * 2))
            self.assertEqual(int(enlarged.sum()), 100)
            self.assertTrue(np.array_equal(resize_mask(enlarged, shape), mask))

    def test_zero_evidence_exact(self):
        a = np.random.default_rng(1).random((256, 256), dtype=np.float32)
        self.assertTrue(np.array_equal(a, fuse_with_evidence(a, np.zeros_like(a), .42)))

    def test_global_ignores_location(self):
        a = component('a', center=(0., 0.))
        b = component('b', center=(1., 1.))
        self.assertEqual(component_cost(a, b, global_query=True), 0.)
        self.assertAlmostEqual(component_cost(a, b), .15)


if __name__ == '__main__':
    unittest.main(verbosity=2)
