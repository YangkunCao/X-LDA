import os
import pandas as pd
import random
from sklearn.utils import shuffle

random.seed(1)
current_path = os.path.dirname(__file__)


class Task():
    def __init__(self):
        self.A, self.Sd, self.Lm, self.Md = self.read_data_flies()
        self.sim_seq_cos = pd.read_excel(current_path + '/data_create/sequence/Lnc_seq_sim_cosine.xlsx', index_col=0,
                                         header=0)
        self.lnc_dis_ass_Pairs = pd.read_excel(current_path + "/data_create/cv/lnc_dis_ass_pairs.xlsx")
        self.mi_sim = pd.read_excel(current_path + '/data_create/miRNA_similaritys_index.xlsx', index_col=0, header=0)
        self.select_zero()

    def read_data_flies(self):

        A = pd.read_excel(current_path + "/data_create/lnc_dis_association_index.xlsx", header=0, index_col=0)

        Sd = pd.read_excel(current_path + "/data_create/dis_sim_matrix_process_index.xlsx", header=0, index_col=0)

        Lm = pd.read_excel(current_path + "/data_create/y_lnc_mi_association.xlsx", header=0, index_col=0)

        Md = pd.read_excel(current_path + "/data_create/mi_dis_index.xlsx", header=0, index_col=0)

        return A, Sd, Lm, Md

    def select_zero(self):
        lnc_dis_ass_NoPairs = []
        for i in self.A.index:
            for j in self.A.columns:
                if self.A.loc[i, j] == 0:
                    lnc_dis_ass_NoPairs.append([i, j])

        random.shuffle(lnc_dis_ass_NoPairs)
        self.lnc_dis_ass_NoPairs = pd.DataFrame(data=lnc_dis_ass_NoPairs, columns=['pair1', 'pair2'])
        self.lnc_dis_ass_NoPairs['ass'] = 0


class TaskA(Task):
    def __init__(self):
        super(TaskA, self).__init__()
        self.cv_A = []
        self.cv_func_sim_lnc = []
        self.test_pairs = []
        self.train_pairs = []
        self.test_real_ass = []
        self.reconstruct_association_similarity()

    def reconstruct_association_similarity(self):
        A_1 = pd.read_excel(current_path + '/data_create/cv/task1/cv1/lnc_dis_ass1.xlsx', index_col=0, header=0)
        A_2 = pd.read_excel(current_path + '/data_create/cv/task1/cv2/lnc_dis_ass2.xlsx', index_col=0, header=0)
        A_3 = pd.read_excel(current_path + '/data_create/cv/task1/cv3/lnc_dis_ass3.xlsx', index_col=0, header=0)
        A_4 = pd.read_excel(current_path + '/data_create/cv/task1/cv4/lnc_dis_ass4.xlsx', index_col=0, header=0)
        A_5 = pd.read_excel(current_path + '/data_create/cv/task1/cv5/lnc_dis_ass5.xlsx', index_col=0, header=0)

        self.cv_A.append(A_1)
        self.cv_A.append(A_2)
        self.cv_A.append(A_3)
        self.cv_A.append(A_4)
        self.cv_A.append(A_5)

        s1 = pd.read_excel(current_path + '/data_create/cv/task1/cv1/sim_lnc_func1.xlsx', index_col=0, header=0)
        s2 = pd.read_excel(current_path + '/data_create/cv/task1/cv2/sim_lnc_func2.xlsx', index_col=0, header=0)
        s3 = pd.read_excel(current_path + '/data_create/cv/task1/cv3/sim_lnc_func3.xlsx', index_col=0, header=0)
        s4 = pd.read_excel(current_path + '/data_create/cv/task1/cv4/sim_lnc_func4.xlsx', index_col=0, header=0)
        s5 = pd.read_excel(current_path + '/data_create/cv/task1/cv5/sim_lnc_func5.xlsx', index_col=0, header=0)

        self.cv_func_sim_lnc.append(s1)
        self.cv_func_sim_lnc.append(s2)
        self.cv_func_sim_lnc.append(s3)
        self.cv_func_sim_lnc.append(s4)
        self.cv_func_sim_lnc.append(s5)

        r2z_pairs1 = pd.read_excel(current_path + '/data_create/cv/task1/cv1/real2zero_pair1.xlsx')
        r2z_pairs2 = pd.read_excel(current_path + '/data_create/cv/task1/cv2/real2zero_pair2.xlsx')
        r2z_pairs3 = pd.read_excel(current_path + '/data_create/cv/task1/cv3/real2zero_pair3.xlsx')
        r2z_pairs4 = pd.read_excel(current_path + '/data_create/cv/task1/cv4/real2zero_pair4.xlsx')
        r2z_pairs5 = pd.read_excel(current_path + '/data_create/cv/task1/cv5/real2zero_pair5.xlsx')

        ### 关联对 去掉 测试集，则为训练集
        train_pairs1 = pd.concat([self.lnc_dis_ass_Pairs, r2z_pairs1], axis=0).drop_duplicates(keep=False)
        train_pairs2 = pd.concat([self.lnc_dis_ass_Pairs, r2z_pairs2], axis=0).drop_duplicates(keep=False)
        train_pairs3 = pd.concat([self.lnc_dis_ass_Pairs, r2z_pairs3], axis=0).drop_duplicates(keep=False)
        train_pairs4 = pd.concat([self.lnc_dis_ass_Pairs, r2z_pairs4], axis=0).drop_duplicates(keep=False)
        train_pairs5 = pd.concat([self.lnc_dis_ass_Pairs, r2z_pairs5], axis=0).drop_duplicates(keep=False)
        train_pairs1['ass'] = 1
        train_pairs2['ass'] = 1
        train_pairs3['ass'] = 1
        train_pairs4['ass'] = 1
        train_pairs5['ass'] = 1

        ### contrust the train set which contains (0, 1)
        train_pairs1 = pd.concat([train_pairs1, self.lnc_dis_ass_NoPairs.iloc[:train_pairs1.shape[0], :]], axis=0)
        train_pairs1 = shuffle(train_pairs1)

        train_pairs2 = pd.concat([train_pairs2, self.lnc_dis_ass_NoPairs.iloc[
                                                train_pairs1.shape[0]: train_pairs1.shape[0] + train_pairs2.shape[0],
                                                :]], axis=0)
        train_pairs2 = shuffle(train_pairs2)

        train_pairs3 = pd.concat([train_pairs3, self.lnc_dis_ass_NoPairs.iloc[
                                                train_pairs1.shape[0] + train_pairs2.shape[0]:train_pairs1.shape[0] +
                                                                                              train_pairs2.shape[0] +
                                                                                              train_pairs3.shape[0],
                                                :]], axis=0)
        train_pairs3 = shuffle(train_pairs3)

        train_pairs4 = pd.concat([train_pairs4, self.lnc_dis_ass_NoPairs.iloc[
                                                train_pairs1.shape[0] + train_pairs2.shape[0] + train_pairs3.shape[0]:
                                                train_pairs1.shape[0] + train_pairs2.shape[0] + train_pairs3.shape[0] +
                                                train_pairs4.shape[0], :]], axis=0)
        train_pairs4 = shuffle(train_pairs4)

        train_pairs5 = pd.concat([train_pairs5, self.lnc_dis_ass_NoPairs.iloc[
                                                train_pairs1.shape[0] + train_pairs2.shape[0] + train_pairs3.shape[0] +
                                                train_pairs4.shape[0]:train_pairs1.shape[0] + train_pairs2.shape[0] +
                                                                      train_pairs3.shape[0] + train_pairs4.shape[0] +
                                                                      train_pairs5.shape[0], :]], axis=0)
        train_pairs5 = shuffle(train_pairs5)

        self.train_pairs.append(train_pairs1)
        self.train_pairs.append(train_pairs2)
        self.train_pairs.append(train_pairs3)
        self.train_pairs.append(train_pairs4)
        self.train_pairs.append(train_pairs5)

        ## 测试集
        r2z_pairs1['ass'] = 1
        r2z_pairs2['ass'] = 1
        r2z_pairs3['ass'] = 1
        r2z_pairs4['ass'] = 1
        r2z_pairs5['ass'] = 1

        self.test_real_ass.append(r2z_pairs1)
        self.test_real_ass.append(r2z_pairs2)
        self.test_real_ass.append(r2z_pairs3)
        self.test_real_ass.append(r2z_pairs4)
        self.test_real_ass.append(r2z_pairs5)

        r2z_pairs1 = pd.concat([r2z_pairs1, self.lnc_dis_ass_NoPairs.iloc[train_pairs1.shape[0]:, :]], axis=0)
        r2z_pairs1 = shuffle(r2z_pairs1)

        tmp_2 = pd.concat([self.lnc_dis_ass_NoPairs.iloc[:train_pairs1.shape[0], :],
                           self.lnc_dis_ass_NoPairs.iloc[train_pairs1.shape[0] + train_pairs2.shape[0]:, :]], axis=0)
        r2z_pairs2 = pd.concat([r2z_pairs2, tmp_2], axis=0)
        r2z_pairs2 = shuffle(r2z_pairs2)

        tmp_3 = pd.concat([self.lnc_dis_ass_NoPairs.iloc[:train_pairs1.shape[0] + train_pairs2.shape[0], :],
                           self.lnc_dis_ass_NoPairs.iloc[
                           train_pairs1.shape[0] + train_pairs2.shape[0] + train_pairs3.shape[0]:, :]], axis=0)
        r2z_pairs3 = pd.concat([r2z_pairs3, tmp_3], axis=0)
        r2z_pairs3 = shuffle(r2z_pairs3)

        tmp_4 = pd.concat(
            [self.lnc_dis_ass_NoPairs.iloc[:train_pairs1.shape[0] + train_pairs2.shape[0] + train_pairs3.shape[0], :],
             self.lnc_dis_ass_NoPairs.iloc[
             train_pairs1.shape[0] + train_pairs2.shape[0] + train_pairs3.shape[0] + train_pairs4.shape[0]:, :]],
            axis=0)
        r2z_pairs4 = pd.concat([r2z_pairs4, tmp_4], axis=0)
        r2z_pairs4 = shuffle(r2z_pairs4)

        tmp_5 = pd.concat(
            [self.lnc_dis_ass_NoPairs.iloc[
             :train_pairs1.shape[0] + train_pairs2.shape[0] + train_pairs3.shape[0] + train_pairs4.shape[0], :],
             self.lnc_dis_ass_NoPairs.iloc[
             train_pairs1.shape[0] + train_pairs2.shape[0] + train_pairs3.shape[0] + train_pairs4.shape[0] +
             train_pairs5.shape[0]:, :]],
            axis=0)
        r2z_pairs5 = pd.concat([r2z_pairs5, tmp_5], axis=0)
        r2z_pairs5 = shuffle(r2z_pairs5)

        self.test_pairs.append(r2z_pairs1)
        self.test_pairs.append(r2z_pairs2)
        self.test_pairs.append(r2z_pairs3)
        self.test_pairs.append(r2z_pairs4)
        self.test_pairs.append(r2z_pairs5)

    def get_pair(self, index):
        return self.train_pairs[index], self.test_pairs[index]

    def get_k_A(self, index):
        return self.cv_A[index]

    def get_k_Sim(self, index):
        return self.cv_func_sim_lnc[index]

    def get_association_similarity_martix(self):
        return self.A, self.Sd, self.Lm, self.Md, self.sim_seq_cos
