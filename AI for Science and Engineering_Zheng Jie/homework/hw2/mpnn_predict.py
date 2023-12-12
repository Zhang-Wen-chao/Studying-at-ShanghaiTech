# 创建模型
mpnn = MPNNModel(
    atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
)

# 加载保存的权重
mpnn.load_weights('path_to_save_model_weights.h5')
