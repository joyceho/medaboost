import numpy as np
import numpy.testing as npt
import pandas as pd

from medaboost.truth import DS


y = pd.DataFrame.from_dict({'0': {'0':0, '1':0, '2':1, '3':1, '4':1},
                            '1': {'0':0, '1':1, '2':0, '3':1, '4':0},
                            '2': {'0':1, '1':0, '2':0, '3':0, '4':0},
                            '3': {'0':0, '1':0, '2':1, '3':0, '4':1},
                            '4': {'0':0, '1':0, '2':0, '3':1, '4':0},
                            '5': {'0':1, '1':0, '2':1, '3':1, '4':1},
                            '6': {'0':0, '1':0, '2':0, '3':0, '4':1},
                            '7': {'0':1, '1':0, '2':0, '3':1, '4':1},
                            '8': {'0':0, '1':0, '2':0, '3':0, '4':0},
                            '9': {'0':1, '1':1, '2':0, '3':1, '4':0},
                            '10': {'0':0, '1':0, '2':0, '3':0, '4':0},
                            '11': {'0':0, '1':1, '2':1, '3':0, '4':1},
                            '12': {'0':1, '1':0, '2':1, '3':0, '4':0},
                            '13': {'0':1, '1':0, '2':1, '3':1, '4':1},
                            '14': {'0':0, '1':1, '2':0, '3':1, '4':0},
                            '15': {'0':1, '1':0, '2':0, '3':0, '4':1},
                            '16': {'0':1, '1':1, '2':1, '3':1, '4':0}, 
                            '17': {'0':1, '1':0, '2':0, '3':1, '4':0}, 
                            '18': {'0':0, '1':0, '2':1, '3':1, '4':1}, 
                            '19': {'0':0, '1':0, '2':0, '3':0, '4':1}, 
                            '20': {'0':0, '1':1, '2':1, '3':1, '4':0}, 
                            '21': {'0':0, '1':0, '2':0, '3':0, '4':0}, 
                            '22': {'0':0, '1':0, '2':0, '3':1, '4':0}, 
                            '23': {'0':0, '1':1, '2':0, '3':0, '4':1}, 
                            '24': {'0':0, '1':1, '2':1, '3':1, '4':0}, 
                            '25': {'0':1, '1':1, '2':1, '3':0, '4':1}, 
                            '26': {'0':0, '1':1, '2':0, '3':1, '4':1}, 
                            '27': {'0':1, '1':1, '2':1, '3':0, '4':1}, 
                            '28': {'0':1, '1':0, '2':0, '3':1, '4':1}, 
                            '29': {'0':0, '1':0, '2':0, '3':1, '4':0}, 
                            '30': {'0':0, '1':1, '2':0, '3':0, '4':1}, 
                            '31': {'0':1, '1':0, '2':1, '3':0, '4':1}, 
                            '32': {'0':1, '1':0, '2':0, '3':0, '4':1}, 
                            '33': {'0':1, '1':0, '2':0, '3':1, '4':0}, 
                            '34': {'0':0, '1':1, '2':1, '3':1, '4':1}, 
                            '35': {'0':1, '1':0, '2':0, '3':1, '4':0}, 
                            '36': {'0':1, '1':0, '2':0, '3':1, '4':0}, 
                            '37': {'0':1, '1':1, '2':1, '3':0, '4':1}, 
                            '38': {'0':1, '1':0, '2':0, '3':1, '4':1}, 
                            '39': {'0':1, '1':0, '2':1, '3':1, '4':1}, 
                            '40': {'0':0, '1':1, '2':1, '3':0, '4':1}, 
                            '41': {'0':1, '1':0, '2':0, '3':1, '4':0}, 
                            '42': {'0':1, '1':0, '2':0, '3':0, '4':1}, 
                            '43': {'0':0, '1':0, '2':0, '3':0, '4':0}, 
                            '44': {'0':1, '1':0, '2':0, '3':0, '4':1}, 
                            '45': {'0':0, '1':0, '2':0, '3':0, '4':0}, 
                            '46': {'0':1, '1':1, '2':1, '3':0, '4':0}, 
                            '47': {'0':0, '1':0, '2':0, '3':0, '4':0}, 
                            '48': {'0':1, '1':1, '2':1, '3':0, '4':1}, 
                            '49': {'0':1, '1':1, '2':1, '3':1, '4':1}},
                            dtype=pd.Int64Dtype(),
                            orient='index')
y = y.to_numpy(dtype='int')


e2lpd = pd.DataFrame.from_dict({'0': {'0': 0.30000000000000004, '1': 0.7},
        '1': {'0': 0.7, '1': 0.30000000000000004},
        '2': {'0': 0.9270270270270269, '1': 0.07297297297297302},
        '3': {'0': 0.7, '1': 0.30000000000000004},
        '4': {'0': 0.9270270270270269, '1': 0.07297297297297302},
        '5': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '6': {'0': 0.9270270270270269, '1': 0.07297297297297302},
        '7': {'0': 0.30000000000000004, '1': 0.7},
        '8': {'0': 0.9857478005865103, '1': 0.014252199413489751},
        '9': {'0': 0.30000000000000004, '1': 0.7},
        '10': {'0': 0.9857478005865103, '1': 0.014252199413489751},
        '11': {'0': 0.30000000000000004, '1': 0.7},
        '12': {'0': 0.7, '1': 0.30000000000000004},
        '13': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '14': {'0': 0.7, '1': 0.30000000000000004},
        '15': {'0': 0.7, '1': 0.30000000000000004},
        '16': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '17': {'0': 0.7, '1': 0.30000000000000004},
        '18': {'0': 0.30000000000000004, '1': 0.7},
        '19': {'0': 0.9270270270270269, '1': 0.07297297297297302},
        '20': {'0': 0.30000000000000004, '1': 0.7},
        '21': {'0': 0.9857478005865103, '1': 0.014252199413489751},
        '22': {'0': 0.9270270270270269, '1': 0.07297297297297302},
        '23': {'0': 0.7, '1': 0.30000000000000004},
        '24': {'0': 0.30000000000000004, '1': 0.7},
        '25': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '26': {'0': 0.30000000000000004, '1': 0.7},
        '27': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '28': {'0': 0.30000000000000004, '1': 0.7},
        '29': {'0': 0.9270270270270269, '1': 0.07297297297297302},
        '30': {'0': 0.7, '1': 0.30000000000000004},
        '31': {'0': 0.30000000000000004, '1': 0.7},
        '32': {'0': 0.7, '1': 0.30000000000000004},
        '33': {'0': 0.7, '1': 0.30000000000000004},
        '34': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '35': {'0': 0.7, '1': 0.30000000000000004},
        '36': {'0': 0.7, '1': 0.30000000000000004},
        '37': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '38': {'0': 0.30000000000000004, '1': 0.7},
        '39': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '40': {'0': 0.30000000000000004, '1': 0.7},
        '41': {'0': 0.7, '1': 0.30000000000000004},
        '42': {'0': 0.7, '1': 0.30000000000000004},
        '43': {'0': 0.9857478005865103, '1': 0.014252199413489751},
        '44': {'0': 0.7, '1': 0.30000000000000004},
        '45': {'0': 0.9857478005865103, '1': 0.014252199413489751},
        '46': {'0': 0.30000000000000004, '1': 0.7},
        '47': {'0': 0.9857478005865103, '1': 0.014252199413489751},
        '48': {'0': 0.07297297297297302, '1': 0.9270270270270269},
        '49': {'0': 0.014252199413489751, '1': 0.9857478005865103}},
         orient='index')
e2lpd = e2lpd.to_numpy()

# test a sparse matrix
y_sparse = pd.DataFrame.from_dict({0: {'0':0, '1':0, '2':1, '3':1, '4':1},
                                   1: {'0':0, '1':1, '2':0, '3':1, '4':0},
                                   2: {'3':0, '4':0},
                                   3: {'1':0, '2':1, '3':0, '4':1}, 
                                   4: {'0':0, '2':0, '3':1},
                                   5: {'2':1, '3':1, '4':1},
                                   6: {'0':0, '1':0, '2':0, '3':0, '4':1},
                                   7: {'0':1, '3':1}, 
                                   8: {'1':0, '3':0, '4':0}, 
                                   9: {'0':1, '1':1, '3':1, '4':0}, 
                                   10: {'0':0, '1':0, '2':0, '3':0},
                                   11: {'0':0, '1':1, '2':1, '4':1},
                                   12: {'0':1, '1':0, '2':1, '3':0, '4':0},
                                   13: {'0':1, '1':0, '2':1},
                                   14: {'0':0},
                                   15: {'0':1, '1':0, '2':0},
                                   16: {'0':1, '1':1, '4':0},
                                   17: {'1':0, '4':0},
                                   18: {'0':0, '1':0, '2':1, '4':1},
                                   19: {'0':0, '1':0, '3':0, '4':1},
                                   20: {'2':1, '3':1, '4':0},
                                   21: {'1':0, '2':0, '3':0, '4':0},
                                   22: {'3':1, '4':0},
                                   23: {'2':0},
                                   24: {'0':0, '1':1, '3':1, '4':0},
                                   25: {'0':1, '1':1, '2':1, '3':0, '4':1},
                                   26: {'0':0, '1':1, '2':0, '3':1},
                                   27: {'0':1, '1':1, '3':0, '4':1},
                                   28: {'0':1, '2':0, '3':1, '4':1},
                                   29: {'1':0, '2':0, '3':1},
                                   30: {'0':0, '1':1, '2':0, '3':0},
                                   31: {'0':1, '1':0, '2':1, '3':0},
                                   32: {'0':1, '3':0, '4':1},
                                   33: {'0':1, '1':0, '2':0, '3':1, '4':0},
                                   34: {'1':1, '2':1, '3':1, '4':1},
                                   35: {'0':1, '2':0, '3':1},
                                   36: {'0':1, '1':0, '2':0, '4':0},
                                   37: {'1':1, '3':0, '4':1},
                                   38: {'1':0, '2':0, '3':1, '4':1},
                                   39: {'0':1, '3':1},
                                   40: {'0':0, '1':1, '3':0, '4':1},
                                   41: {'0':1, '2':0, '3':1},
                                   42: {'0':1, '1':0, '2':0, '3':0, '4':1},
                                   43: {'0':0, '1':0, '2':0, '3':0},
                                   44: {'0':1, '2':0, '3':0},
                                   45: {'0':0, '1':0, '2':0, '3':0, '4':0},
                                   46: {'0':1, '1':1, '2':1, '3':0, '4':0},
                                   47: {'0':0, '4':0},
                                   48: {'0':1, '1':1, '3':0, '4':1},
                                   49: {'0':1, '2':1, '3':1, '4':1}},
                        dtype=pd.Int64Dtype(),
                        orient='index')
y_sparse = y_sparse.sort_index().to_numpy(dtype='float', na_value=np.nan)


e2lpd_sparse = pd.DataFrame.from_dict({'0': {'0': 0.30000000000000004, '1': 0.7},
                                       '1': {'0': 0.7, '1': 0.30000000000000004},
                                       '2': {'0': 0.8448275862068965,'1': 0.1551724137931035},
                                       '3': {'0': 0.5, '1': 0.5}, 
                                       '4': {'0': 0.6999999999999998, '1': 0.30000000000000004}, 
                                       '5': {'0': 0.07297297297297302, '1': 0.927027027027027},
                                       '6': {'0': 0.9270270270270269, '1': 0.07297297297297302}, 
                                       '7': {'0': 0.1551724137931035, '1': 0.8448275862068965}, 
                                       '8': {'0': 0.927027027027027, '1': 0.07297297297297302}, 
                                       '9': {'0': 0.1551724137931035, '1': 0.8448275862068966}, 
                                       '10': {'0': 0.967365028203062, '1': 0.03263497179693798}, 
                                       '11': {'0': 0.1551724137931035, '1': 0.8448275862068966}, 
                                       '12': {'0': 0.7, '1': 0.30000000000000004}, 
                                       '13': {'0': 0.30000000000000004, '1': 0.6999999999999998}, 
                                       '14': {'0': 0.7, '1': 0.30000000000000004}, 
                                       '15': {'0': 0.6999999999999998, '1': 0.30000000000000004}, 
                                       '16': {'0': 0.30000000000000004, '1': 0.6999999999999998}, 
                                       '17': {'0': 0.8448275862068965, '1': 0.1551724137931035}, 
                                       '18': {'0': 0.5, '1': 0.5}, 
                                       '19': {'0': 0.8448275862068966, '1': 0.1551724137931035}, 
                                       '20': {'0': 0.30000000000000004, '1': 0.6999999999999998}, 
                                       '21': {'0': 0.967365028203062, '1': 0.03263497179693798}, 
                                       '22': {'0': 0.5, '1': 0.5}, 
                                       '23': {'0': 0.7, '1': 0.30000000000000004}, 
                                       '24': {'0': 0.5, '1': 0.5}, 
                                       '25': {'0': 0.07297297297297302, '1': 0.9270270270270269}, 
                                       '26': {'0': 0.5, '1': 0.5}, 
                                       '27': {'0': 0.1551724137931035, '1': 0.8448275862068966}, 
                                       '28': {'0': 0.1551724137931035, '1': 0.8448275862068966}, 
                                       '29': {'0': 0.6999999999999998, '1': 0.30000000000000004}, 
                                       '30': {'0': 0.8448275862068966, '1': 0.1551724137931035}, 
                                       '31': {'0': 0.5, '1': 0.5}, 
                                       '32': {'0': 0.30000000000000004, '1': 0.6999999999999998}, 
                                       '33': {'0': 0.7, '1': 0.30000000000000004}, 
                                       '34': {'0': 0.03263497179693798, '1': 0.967365028203062}, 
                                       '35': {'0': 0.30000000000000004, '1': 0.6999999999999998}, 
                                       '36': {'0': 0.8448275862068966, '1': 0.1551724137931035}, 
                                       '37': {'0': 0.30000000000000004, '1': 0.6999999999999998}, 
                                       '38': {'0': 0.5, '1': 0.5}, 
                                       '39': {'0': 0.1551724137931035, '1': 0.8448275862068965}, 
                                       '40': {'0': 0.5, '1': 0.5}, 
                                       '41': {'0': 0.30000000000000004, '1': 0.6999999999999998}, 
                                       '42': {'0': 0.7, '1': 0.30000000000000004}, 
                                       '43': {'0': 0.967365028203062, '1': 0.03263497179693798}, 
                                       '44': {'0': 0.6999999999999998, '1': 0.30000000000000004}, 
                                       '45': {'0': 0.9857478005865103, '1': 0.014252199413489751}, 
                                       '46': {'0': 0.30000000000000004, '1': 0.7}, 
                                       '47': {'0': 0.8448275862068965, '1': 0.1551724137931035}, 
                                       '48': {'0': 0.1551724137931035, '1': 0.8448275862068966}, 
                                       '49': {'0': 0.03263497179693798, '1': 0.967365028203062}},
                                   orient='index')


def setup_model(y_prob):
    model = DS()
    # setup the worker confidence interval
    cm = np.full((2, 2), 0.3)
    np.fill_diagonal(cm, 0.7)
    model.w2cm = np.repeat(cm[:, :, np.newaxis], 5, axis=2)
    model.l2pd = np.full(2, 1/2)
    model.n_labels = 2
    model.n_tasks = y_prob.shape[0]
    model.n_workers = y_prob.shape[1]
    model.y_mat = y_prob
    return model


def test_e_step():
    mv = setup_model(y)
    mv.e_step()
    npt.assert_array_almost_equal(mv.e2lpd, e2lpd)


def test_e_step_sparse():
    mv = setup_model(y_sparse)
    mv.e_step()
    npt.assert_array_almost_equal(mv.e2lpd, e2lpd_sparse)


def test_m_step_l2pd():
    mv = DS()
    tmp_e2lpd = pd.DataFrame.from_dict({'0': {'0': 0.243329342489206,'1': 0.756670657510794},
                     '1': {'0': 0.7416227366627413, '1': 0.2583772633372587},
                     '2': {'0': 0.9050316844306507, '1': 0.09496831556934929},
                     '3': {'0': 0.4217301646530141, '1': 0.5782698353469858},
                     '4': {'0': 0.9301022064787614, '1': 0.06989779352123852},
                     '5': {'0': 0.09219000773339762, '1': 0.9078099922666024},
                     '6': {'0': 0.8758958292757671, '1': 0.1241041707242328},
                     '7': {'0': 0.4956544677522057, '1': 0.5043455322477943},
                     '8': {'0': 0.9679256221719688, '1': 0.03207437782803112},
                     '9': {'0': 0.4754569397551008, '1': 0.5245430602448992},
                     '10': {'0': 0.9679256221719688, '1': 0.03207437782803112},
                     '11': {'0': 0.13592980278392428, '1': 0.8640701972160757},
                     '12': {'0': 0.49615653973757506, '1': 0.5038434602624249},
                     '13': {'0': 0.09219000773339762, '1': 0.9078099922666024},
                     '14': {'0': 0.7416227366627413, '1': 0.2583772633372587},
                     '15': {'0': 0.6902855847451727, '1': 0.3097144152548274},
                     '16': {'0': 0.08564155982002945, '1': 0.9143584401799706},
                     '17': {'0': 0.8077706998809536, '1': 0.1922293001190464},
                     '18': {'0': 0.243329342489206, '1': 0.756670657510794},
                     '19': {'0': 0.8758958292757671, '1': 0.1241041707242328},
                     '20': {'0': 0.22875035005821284, '1': 0.7712496499417871},
                     '21': {'0': 0.9679256221719688, '1': 0.03207437782803112},
                     '22': {'0': 0.9301022064787614, '1': 0.06989779352123852},
                     '23': {'0': 0.6035515334077811, '1': 0.39644846659221883},
                     '24': {'0': 0.22875035005821284, '1': 0.7712496499417871},
                     '25': {'0': 0.047327191525097394, '1': 0.9526728084749027},
                     '26': {'0': 0.4016602938687415, '1': 0.5983397061312585},
                     '27': {'0': 0.047327191525097394, '1': 0.9526728084749027},
                     '28': {'0': 0.4956544677522057, '1': 0.5043455322477943},
                     '29': {'0': 0.9301022064787614, '1': 0.06989779352123852},
                     '30': {'0': 0.6035515334077811, '1': 0.39644846659221883},
                     '31': {'0': 0.18719414239129673, '1': 0.8128058576087032},
                     '32': {'0': 0.6902855847451727, '1': 0.3097144152548274},
                     '33': {'0': 0.8077706998809536, '1': 0.1922293001190464},
                     '34': {'0': 0.0648668175218792, '1': 0.9351331824781208},
                     '35': {'0': 0.8077706998809536, '1': 0.1922293001190464},
                     '36': {'0': 0.8077706998809536, '1': 0.1922293001190464},
                     '37': {'0': 0.047327191525097394, '1': 0.9526728084749027},
                     '38': {'0': 0.4956544677522057, '1': 0.5043455322477943},
                     '39': {'0': 0.09219000773339762, '1': 0.9078099922666024},
                     '40': {'0': 0.13592980278392428, '1': 0.8640701972160757},
                     '41': {'0': 0.8077706998809536, '1': 0.1922293001190464},
                     '42': {'0': 0.6902855847451727, '1': 0.3097144152548274},
                     '43': {'0': 0.9679256221719688, '1': 0.03207437782803112},
                     '44': {'0': 0.6902855847451727, '1': 0.3097144152548274},
                     '45': {'0': 0.9679256221719688, '1': 0.03207437782803112},
                     '46': {'0': 0.1751997266202346, '1': 0.8248002733797654},
                     '47': {'0': 0.9679256221719688, '1': 0.03207437782803112},
                     '48': {'0': 0.047327191525097394, '1': 0.9526728084749027},
                     '49': {'0': 0.021435802585586, '1': 0.978564197414414}},
                     orient='index')
    mv.e2lpd = tmp_e2lpd.to_numpy()
    mv.lpd = np.full(2, 1/2)
    mv.n_tasks = mv.e2lpd.shape[0]
    # run the m-step
    mv.m_step_l2pd()
    npt.assert_array_almost_equal(mv.l2pd, np.array([0.5248646248830027, 0.4751353751169974]))


def test_m_step_w2cm():
    w2cm = np.zeros((2, 2, 5))
    w2cm[:, :, 0] = np.array([[0.6110744292163828, 0.38892557078361717],
                              [0.33162594432815073, 0.6683740556718494]])
    w2cm[:, :, 1] = np.array([[0.7983969036739015, 0.2016030963260983],
                              [0.4606973614574203, 0.5393026385425799]])
    w2cm[:, :, 2] = np.array([[0.8315855594745271, 0.1684144405254727],
                              [0.33784903051116916, 0.6621509694888308]])
    w2cm[:, :, 3] = np.array([[0.5946837127013499, 0.40531628729865016],
                              [0.3928196251278312, 0.6071803748721689]])
    w2cm[:, :, 4] = np.array([[0.6213568041730184, 0.37864319582698164],
                              [0.2773468037318241, 0.7226531962681759]])

    mv = DS()
    mv.n_workers = 5
    mv.n_labels = 2
    mv.y_mat = y
    mv.e2lpd = e2lpd
    # run the m step
    mv.m_step_w2cm()
    npt.assert_array_almost_equal(mv.w2cm, w2cm)


def test_m_step_w2cm_sparse():
    w2cm = np.zeros((2, 2, 5))
    w2cm[:, :, 0] = np.array([[0.5874309772101427, 0.4125690227898573],
                              [0.27543322194664427, 0.7245667780533559]])
    w2cm[:, :, 1] = np.array([[0.7547644578861261, 0.24523554],
                              [0.37609912315700195, 0.623900876842998]])
    w2cm[:, :, 2] = np.array([[0.7863173738235009, 0.21368262617649922],
                              [0.39939747434484424, 0.6006025256551557]])
    w2cm[:, :, 3] = np.array([[0.6606728199716017, 0.33932718002839835],
                              [0.3904266755189602, 0.6095733244810396]])
    w2cm[:, :, 4] = np.array([[0.6266929073113172, 0.373307092688683],
                              [0.32133818806505865, 0.6786618119349415]])

    mv = DS()
    mv.n_workers = 5
    mv.n_labels = 2
    mv.y_mat = y_sparse
    mv.e2lpd = e2lpd_sparse.to_numpy()
    # run the m step
    mv.m_step_w2cm()
    npt.assert_array_almost_equal(mv.w2cm, w2cm)


def test_infer():
    # try the regular y
    mv = DS()
    _ = mv.infer(y)
    # verify w2cm
    w2cm = np.zeros((2, 2, 5))
    w2cm[:, :, 0] = np.array([[0.5247423930566665, 0.4752576069433335],
                              [0.42725069075304634, 0.5727493092469539]])
    w2cm[:, :, 1] = np.array([[0.8387310366082579, 0.1612689633917424],
                              [0.4057048675572617, 0.5942951324427383]])
    w2cm[:, :, 2] = np.array([[0.9994987276207267, 0.000501272379273483],
                              [0.12900860934414146, 0.8709913906558585]])
    w2cm[:, :, 3] = np.array([[0.4939684036185538, 0.5060315963814461],
                              [0.5071109862714488, 0.49288901372855143]])
    w2cm[:, :, 4] = np.array([[0.6403709970472324, 0.3596290029527676],
                              [0.24735054492110406, 0.7526494550788959]])
    npt.assert_array_almost_equal(mv.w2cm, w2cm)

    # verify e2lpd
    e2lpd = pd.DataFrame.from_dict({'0': {'0': 0.001105137426186952, '1': 0.9988948625738131},
                                    '1': {'0': 0.8868275440654547, '1': 0.11317245593454543},
                                    '2': {'0': 0.9747595399268997, '1': 0.025240460073100372},
                                    '3': {'0': 0.0010473213924117178, '1': 0.9989526786075883},
                                    '4': {'0': 0.9836370587051253, '1': 0.01636294129487469},
                                    '5': {'0': 0.0007494782893864273, '1': 0.9992505217106136},
                                    '6': {'0': 0.9127780503506548, '1': 0.08722194964934522},
                                    '7': {'0': 0.8821682301459726, '1': 0.11783176985402748},
                                    '8': {'0': 0.9827483647143008, '1': 0.017251635285699046},
                                    '9': {'0': 0.8415803689254588, '1': 0.1584196310745411},
                                    '10': {'0': 0.9827483647143008, '1': 0.017251635285699046},
                                    '11': {'0': 0.00013664716699978165, '1': 0.9998633528330003},
                                    '12': {'0': 0.0038540634130024144, '1': 0.9961459365869976},
                                    '13': {'0': 0.0007494782893864273, '1': 0.9992505217106136},
                                    '14': {'0': 0.8868275440654547, '1': 0.11317245593454543},
                                    '15': {'0': 0.8764607629908242, '1': 0.12353923700917577},
                                    '16': {'0': 0.0005319259594899397, '1': 0.99946807404051},
                                    '17': {'0': 0.9760497377617573, '1': 0.023950262238242606},
                                    '18': {'0': 0.001105137426186952, '1': 0.9988948625738131},
                                    '19': {'0': 0.9127780503506548, '1': 0.08722194964934522},
                                    '20': {'0': 0.0007844283608658578, '1': 0.9992155716391342},
                                    '21': {'0': 0.9827483647143008, '1': 0.017251635285699046},
                                    '22': {'0': 0.9836370587051253, '1': 0.01636294129487469},
                                    '23': {'0': 0.5770162273967276, '1': 0.4229837726032723},
                                    '24': {'0': 0.0007844283608658578, '1': 0.9992155716391342},
                                    '25': {'0': 9.264199771209874e-05, '1': 0.9999073580022879},
                                    '26': {'0': 0.5900877875500581, '1': 0.40991221244994186},
                                    '27': {'0': 9.264199771209874e-05, '1': 0.9999073580022879},
                                    '28': {'0': 0.8821682301459726, '1': 0.11783176985402748},
                                    '29': {'0': 0.9836370587051253, '1': 0.01636294129487469},
                                    '30': {'0': 0.5770162273967276, '1': 0.4229837726032723},
                                    '31': {'0': 0.0007102555809087343, '1': 0.9992897444190914},
                                    '32': {'0': 0.8764607629908242, '1': 0.12353923700917577},
                                    '33': {'0': 0.9760497377617573, '1': 0.023950262238242606},
                                    '34': {'0': 0.00014419785533999137, '1': 0.99985580214466},
                                    '35': {'0': 0.9760497377617573, '1': 0.023950262238242606},
                                    '36': {'0': 0.9760497377617573, '1': 0.023950262238242606},
                                    '37': {'0': 9.264199771209874e-05, '1': 0.9999073580022879},
                                    '38': {'0': 0.8821682301459726, '1': 0.11783176985402748},
                                    '39': {'0': 0.0007494782893864273, '1': 0.9992505217106136},
                                    '40': {'0': 0.00013664716699978165, '1': 0.9998633528330003},
                                    '41': {'0': 0.9760497377617573, '1': 0.023950262238242606},
                                    '42': {'0': 0.8764607629908242, '1': 0.12353923700917577},
                                    '43': {'0': 0.9827483647143008, '1': 0.017251635285699046},
                                    '44': {'0': 0.8764607629908242, '1': 0.12353923700917577},
                                    '45': {'0': 0.9827483647143008, '1': 0.017251635285699046},
                                    '46': {'0': 0.0005040827494912571, '1': 0.9994959172505087},
                                    '47': {'0': 0.9827483647143008, '1': 0.017251635285699046},
                                    '48': {'0': 9.264199771209874e-05, '1': 0.9999073580022879},
                                    '49': {'0': 9.776133794307535e-05, '1': 0.9999022386620569}},
                                    orient='index')
    npt.assert_array_almost_equal(mv.e2lpd, e2lpd)


def test_infer_sparse():
    mv = DS()
    _ = mv.infer(y_sparse)
    w2cm = np.zeros((2, 2, 5))
    w2cm[:, :, 0] = np.array([[0.514523474336722, 0.4854765256632782],
                              [0.3364364662822746, 0.6635635337177257]])
    w2cm[:, :, 1] = np.array([[0.6902495923866783, 0.30975040761332157],
                              [0.4796598940955933, 0.5203401059044069]])
    w2cm[:, :, 2] = np.array([[0.9708762902969136, 0.029123709703086663],
                              [0.16742334903386474, 0.8325766509661355]])
    w2cm[:, :, 3] = np.array([[0.4631512669363408, 0.5368487330636593],
                              [0.5957881757241581, 0.40421182427584224]])
    w2cm[:, :, 4] = np.array([[0.7027659354117682, 0.2972340645882317],
                              [0.27529291089392166, 0.7247070891060785]])
    npt.assert_array_almost_equal(mv.w2cm, w2cm)
    e2lpd = pd.DataFrame.from_dict({'0': {'0': 0.050703165257977016, '1': 0.949296834742023},
                                    '1': {'0': 0.9530093888408384, '1': 0.04699061115916151}, 
                                    '2': {'0': 0.699335650534152, '1': 0.300664349465848}, 
                                    '3': {'0': 0.02001538561006869, '1': 0.9799846143899313}, 
                                    '4': {'0': 0.9298517299100357, '1': 0.07014827008996428}, 
                                    '5': {'0': 0.02374426048482655, '1': 0.9762557395151734}, 
                                    '6': {'0': 0.8192948184187726, '1': 0.1807051815812274}, 
                                    '7': {'0': 0.5316647567523035, '1': 0.4683352432476965}, 
                                    '8': {'0': 0.7698413074796171, '1': 0.23015869252038282}, 
                                    '9': {'0': 0.6346207456072914, '1': 0.3653792543927086}, 
                                    '10': {'0': 0.9175686917533944, '1': 0.08243130824660558}, 
                                    '11': {'0': 0.01636506483621688, '1': 0.9836349351637831}, 
                                    '12': {'0': 0.08617778055951056, '1': 0.9138222194404895}, 
                                    '13': {'0': 0.04514631965601615, '1': 0.954853680343984}, 
                                    '14': {'0': 0.6403957164852621, '1': 0.3596042835147379}, 
                                    '15': {'0': 0.8729826408926995, '1': 0.1270173591073006}, 
                                    '16': {'0': 0.5663974672260949, '1': 0.4336025327739051}, 
                                    '17': {'0': 0.8115972682925774, '1': 0.18840273170742253}, 
                                    '18': {'0': 0.03861785980515451, '1': 0.9613821401948455}, 
                                    '19': {'0': 0.44748665376945834, '1': 0.5525133462305416}, 
                                    '20': {'0': 0.1329912154932963, '1': 0.8670087845067037}, 
                                    '21': {'0': 0.9493011742039275, '1': 0.05069882579607259}, 
                                    '22': {'0': 0.7993228015297037, '1': 0.2006771984702963}, 
                                    '23': {'0': 0.8671646643818564, '1': 0.13283533561814373}, 
                                    '24': {'0': 0.7836845702102238, '1': 0.21631542978977628}, 
                                    '25': {'0': 0.006155160786430339, '1': 0.9938448392135696}, 
                                    '26': {'0': 0.8875787457826311, '1': 0.1124212542173689}, 
                                    '27': {'0': 0.13854150141049906, '1': 0.861458498589501}, 
                                    '28': {'0': 0.721326644964483, '1': 0.27867335503551716}, 
                                    '29': {'0': 0.9258293626013065, '1': 0.07417063739869344}, 
                                    '30': {'0': 0.8217593941389686, '1': 0.17824060586103146}, 
                                    '31': {'0': 0.035411793350780374, '1': 0.9645882066492195}, 
                                    '32': {'0': 0.2126061482560077, '1': 0.7873938517439922}, 
                                    '33': {'0': 0.959142471911958, '1': 0.0408575280880419}, 
                                    '34': {'0': 0.014279421101269311, '1': 0.9857205788987307}, 
                                    '35': {'0': 0.864037464895945, '1': 0.1359625351040549}, 
                                    '36': {'0': 0.9463953516359777, '1': 0.05360464836402234}, 
                                    '37': {'0': 0.1801027969802172, '1': 0.8198972030197827}, 
                                    '38': {'0': 0.8356397025645951, '1': 0.1643602974354049}, 
                                    '39': {'0': 0.5316647567523035, '1': 0.4683352432476965}, 
                                    '40': {'0': 0.25118923656112085, '1': 0.7488107634388792}, 
                                    '41': {'0': 0.864037464895945, '1': 0.1359625351040549}, 
                                    '42': {'0': 0.684903688066054, '1': 0.3150963119339461}, 
                                    '43': {'0': 0.9175686917533944, '1': 0.08243130824660558}, 
                                    '44': {'0': 0.7877315842543785, '1': 0.21226841574562147}, 
                                    '45': {'0': 0.9662092990499447, '1': 0.03379070095005529}, 
                                    '46': {'0': 0.037591085105223424, '1': 0.9624089148947765},
                                    '47': {'0': 0.8206142725220142, '1': 0.17938572747798578}, 
                                    '48': {'0': 0.13854150141049906, '1': 0.861458498589501}, 
                                    '49': {'0': 0.01749501376353099, '1': 0.982504986236469}},
                                    orient='index')
    npt.assert_array_almost_equal(mv.e2lpd, e2lpd)