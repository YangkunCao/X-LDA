import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data


def rondom_combine_MetaPaths(l_l: pd.Series, l_d: pd.Series, l_m: pd.Series,
                             d_l: pd.Series, d_d: pd.Series, d_m: pd.Series):
    link_module_r1 = []
    link_module_r2 = []
    assert len(l_l) == 240 and len(l_d) == 405 and len(l_m) == 495
    assert len(d_l) == 240 and len(d_d) == 405 and len(d_m) == 495

    for i in range(len(l_l)):
        link_module_r1.append(l_l[i])
        link_module_r2.append(d_l[i])

        link_module_r1.append(l_d[i])
        link_module_r2.append(d_d[i])

    for i in range(len(l_d) - len(l_l)):
        link_module_r1.append(l_d[240 + i])
        link_module_r2.append(d_d[240 + i])

        link_module_r1.append(l_m[i])
        link_module_r2.append(d_m[i])

    for i in range(240):
        link_module_r1.append(l_m[165 + i])
        link_module_r2.append(d_m[165 + i])

        link_module_r1.append(l_l[i])
        link_module_r2.append(d_l[i])

    for i in range(90):
        link_module_r1.append(l_m[405 + i])
        link_module_r2.append(d_m[405 + i])

        link_module_r1.append(l_d[i])
        link_module_r2.append(d_d[i])

    for i in range(315):
        link_module_r1.append(l_d[90 + i])
        link_module_r2.append(d_d[90 + i])

        link_module_r1.append(l_m[i])
        link_module_r2.append(d_m[i])

    return link_module_r1, link_module_r2


def load_data_XLDA(data: pd.DataFrame, A: pd.DataFrame, Rna_matrix: pd.DataFrame,
                   disease_matrix: pd.DataFrame, BATCH_SIZE, lnc_mi: pd.DataFrame, mi_dis: pd.DataFrame, drop=False,
                   shuffle=True):
    x = []
    y = []
    for j in range(data.shape[0]):
        temp_save = []
        x_A = data['pair1'].iloc[j]
        y_A = data['pair2'].iloc[j]

        link_module_r1, link_module_r2 = rondom_combine_MetaPaths(
            Rna_matrix.loc[x_A, :].values, A.loc[x_A, :].values, lnc_mi.loc[x_A, :].values,
            A.loc[:, y_A].values, disease_matrix.loc[:, y_A].values, mi_dis.loc[:, y_A].values)
        rna_disease_mi = np.concatenate(
            (Rna_matrix.loc[x_A, :].values, A.loc[x_A, :].values, lnc_mi.loc[x_A, :].values,
             Rna_matrix.loc[x_A, :].values, A.loc[x_A, :].values, lnc_mi.loc[x_A, :].values,
             Rna_matrix.loc[x_A, :].values, A.loc[x_A, :].values, lnc_mi.loc[x_A, :].values,
             link_module_r1), axis=0)

        disease_rna_mi = np.concatenate(
            (A.loc[:, y_A].values, disease_matrix.loc[:, y_A].values, mi_dis.loc[:, y_A].values,
             A.loc[:, y_A].values, disease_matrix.loc[:, y_A].values, mi_dis.loc[:, y_A].values,
             A.loc[:, y_A].values, disease_matrix.loc[:, y_A].values, mi_dis.loc[:, y_A].values,
             link_module_r2), axis=0)
        temp_save.append(rna_disease_mi)
        temp_save.append(disease_rna_mi)

        x.append([temp_save])
        y.append(data['ass'].iloc[j])
    x = torch.FloatTensor(np.array(x))

    y = torch.LongTensor(np.array(y))
    torch_dataset = Data.TensorDataset(x, y)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop
    )
    return data_loader
