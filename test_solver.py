import unittest
from solver import valid_grid, predict_task, fit_hypotheses, solve, crop, geometry


class SolverTests(unittest.TestCase):
    def test_grid_domain(self):
        self.assertTrue(valid_grid([[0,9],[3,4]]))
        for grid in ([],[[]],[[True]],[[10]],[[1],[2,3]],[[1]*31]):
            self.assertFalse(valid_grid(grid))

    def test_exactly_two_valid_attempts_without_examples(self):
        result,_=predict_task(dict(train=[],test=[dict(input=[[1,2],[3,4]])]))
        self.assertEqual(set(result[0]),{'attempt_1','attempt_2'})
        self.assertTrue(all(valid_grid(g) for g in result[0].values()))

    def test_geometry_fit(self):
        samples=[[[1,2,3],[4,0,5]],[[6,0],[7,8],[0,9]]]
        train=[dict(input=g,output=geometry(g,1)) for g in samples]
        target=[[0,2,3],[4,5,6]]
        result,_=predict_task(dict(train=train,test=[dict(input=target)]))
        self.assertIn(geometry(target,1),list(result[0].values()))

    def test_color_map_generalization(self):
        train=[dict(input=[[0,1],[1,0]],output=[[0,2],[2,0]]),
               dict(input=[[1,1,0]],output=[[2,2,0]])]
        result,_=predict_task(dict(train=train,test=[dict(input=[[0,0,1],[1,0,0]])]))
        self.assertIn([[0,0,2],[2,0,0]],list(result[0].values()))

    def test_crop_and_empty_foreground(self):
        self.assertEqual(crop([[0,0,0],[0,2,0],[0,0,0]],0),[[2]])
        self.assertEqual(crop([[0]],0),[[0]])

    def test_all_fitted_hypotheses_reproduce_training(self):
        train=[dict(input=[[0,1],[1,0]],output=[[0,2],[2,0]])]
        for h in fit_hypotheses(train):
            self.assertEqual(h.apply(train[0]['input']),train[0]['output'])

    def test_labels_do_not_leak(self):
        train=[dict(input=[[1,2],[3,4]],output=[[1,2],[3,4]])]
        a=dict(train=train,test=[dict(input=[[4,3],[2,1]],output=[[9]])])
        b=dict(train=train,test=[dict(input=[[4,3],[2,1]],output=[[0,0,0]])])
        self.assertEqual(predict_task(a),predict_task(b))

    def test_task_ids_do_not_affect_predictions(self):
        task=dict(train=[],test=[dict(input=[[2]])])
        result=solve({'alpha':task,'unseen':task})
        self.assertEqual(result['alpha'],result['unseen'])

    def test_deterministic_and_no_input_mutation(self):
        task=dict(train=[dict(input=[[0,1]],output=[[0,2]])],test=[dict(input=[[1,0]])])
        import copy
        original=copy.deepcopy(task)
        self.assertEqual(predict_task(task),predict_task(task))
        self.assertEqual(task,original)

    def test_multiple_test_inputs_preserved(self):
        result,_=predict_task(dict(train=[],test=[dict(input=[[2]]),dict(input=[[3,4]])]))
        self.assertEqual(len(result),2)

    def test_invalid_input_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            predict_task(dict(train=[],test=[dict(input=[[12]])]))


if __name__ == '__main__':
    unittest.main()
