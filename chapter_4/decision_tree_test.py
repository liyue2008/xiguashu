#-*-coding:utf-8-*- 
import os
import unittest as ut
import decision_tree as dt
import pandas as pd
import math


class TestDecisionTreeMethods(ut.TestCase):
    def test_all_same(self):
        self.assertTrue(dt.all_same(['好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜']))
        self.assertFalse(dt.all_same(['好瓜', '好瓜', '坏瓜', '好瓜', '好瓜', '好瓜']))
    def test_majority_in_list(self):
        self.assertEqual('好瓜', dt.majority_in_list(['好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜']))
        self.assertEqual('坏瓜', dt.majority_in_list(['好瓜', '坏瓜', '未知', '好瓜', '坏瓜', '坏瓜']))
    def test_TrainingSet_partition_by_attr(self):
        data = {
            '编号': [1, 2, 3, 4],
            '色泽': ['乌黑', '乌黑', '乌黑', '乌黑'],
            '根蒂': ['蜷缩', '硬挺', '蜷缩', '稍蜷'],
            '敲声': ['沉闷', '沉闷', '沉闷', '沉闷'],
            '好瓜': ['是', '是', '是', '否']
        }
        df = pd.DataFrame(data).set_index('编号')
        
        ts = dt.TrainingSet(df, '好瓜')
        a = dt.Attribute('根蒂', {'蜷缩', '硬挺', '稍蜷'})
        

        expect_data = {
            '蜷缩': dt.TrainingSet(pd.DataFrame({
                '编号': [1, 3],
                '色泽': ['乌黑', '乌黑'],
                '根蒂': ['蜷缩', '蜷缩'],
                '敲声': ['沉闷', '沉闷'],
                '好瓜': ['是', '是']
            }).set_index('编号'), '好瓜'),'硬挺': dt.TrainingSet(pd.DataFrame({
                '编号': [2],
                '色泽': ['乌黑'],
                '根蒂': ['硬挺'],
                '敲声': ['沉闷'],
                '好瓜': ['是']
            }).set_index('编号'), '好瓜'),'稍蜷': dt.TrainingSet(pd.DataFrame({
                '编号': [4],
                '色泽': ['乌黑'],
                '根蒂': ['稍蜷'],
                '敲声': ['沉闷'],
                '好瓜': ['否']
            }).set_index('编号'), '好瓜')
        }
    
        sub_ts = ts.partition_by_attr(a)
        self.assertDictEqual(expect_data, sub_ts)
       
    def test_TrainingSet_all_samples_same(self):
        data = {
        '色泽': ['乌黑', '乌黑', '乌黑', '乌黑'],
        '根蒂': ['蜷缩', '硬挺', '蜷缩', '稍蜷'],
        '敲声': ['沉闷', '沉闷', '沉闷', '沉闷'],
        '好瓜': ['是', '是', '是', '否']
        }

        df = pd.DataFrame(data)
        ts = dt.TrainingSet(df, '好瓜')

        a1 = {
            dt.Attribute('色泽', {'青绿', '乌黑', '浅白'}),
            dt.Attribute('根蒂', {'稍蜷', '蜷缩', '硬挺'}),
            dt.Attribute('敲声', {'沉闷', '浊响', '清脆'})
        }

        a2 = {
            dt.Attribute('色泽', {'青绿', '乌黑', '浅白'}),
            dt.Attribute('敲声', {'沉闷', '浊响', '清脆'})
        }

        self.assertFalse(ts.all_samples_same(a1))
        self.assertTrue(ts.all_samples_same(a2))
    def test_ent(self):
        expect_ent = 0.998
        data_file = 'chapter_4/西瓜数据集 2.0.csv'
        df = pd.read_csv(data_file)
        df.set_index('编号', inplace=True)
        D = dt.TrainingSet(df, '好瓜')
        actrual_ent = dt.ent(D)
        # print('actrual_ent = %d\n' % actrual_ent)
        self.assertEqual(expect_ent, round(actrual_ent, 3))
    def test_gain(self):
        print(os.getcwd())
        a = dt.Attribute('色泽', {'青绿', '乌黑', '浅白'})
        expect_gain = 0.108
        data_file = 'chapter_4/西瓜数据集 2.0.csv'
        df = pd.read_csv(data_file)
        df.set_index('编号', inplace=True)
        D = dt.TrainingSet(df, '好瓜')
        actrual_gain = dt.gain(D, a)
        # print(actrual_gain)
        self.assertEqual(expect_gain, round(actrual_gain, 3))
    
if __name__ == '__main__':
    ut.main()