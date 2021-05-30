from keras.models import Model
from keras.layers import Lambda, Dense, Input, Average, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, \
    BatchNormalization, Activation, Dropout, Flatten, AveragePooling2D, Reshape, Conv1D, Add, Subtract,\
    multiply
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import numpy as np
from keras.initializers import glorot_uniform,zeros
from keras.losses import mean_absolute_error, mean_squared_error
import tensorflow as tf


def change_results(third_dense_results_home, third_dense_results_away, inputs3, inputs4, opt, home_pts,
                       home, away):

    home_5 = home[0]
    away_5 = away[0]
    home_6 = home[1]
    away_6 = away[1]
    home_7 = home[2]
    away_7 = away[2]
    home_8 = home[3]
    away_8 = away[3]
    home_9 = home[4]
    away_9 = away[4]
    home_10 = home[5]
    away_10 = away[5]
    home_11 = home[6]
    away_11 = away[6]
    home_12 = home[7]
    away_12 = away[7]
    home_13 = home[8]
    away_13 = away[8]

    home_average_team_embedding5 = multiply([Add()(third_dense_results_home[:5]), inputs3])
    away_average_team_embedding5 = multiply([Add()(third_dense_results_away[:5]), inputs4])
    home_average_team_embedding6 = multiply([Add()(third_dense_results_home[:6]), inputs3])
    away_average_team_embedding6 = multiply([Add()(third_dense_results_away[:6]), inputs4])
    home_average_team_embedding7 = multiply([Add()(third_dense_results_home[:7]), inputs3])
    away_average_team_embedding7 = multiply([Add()(third_dense_results_away[:7]), inputs4])
    home_average_team_embedding8 = multiply([Add()(third_dense_results_home[:8]), inputs3])
    away_average_team_embedding8 = multiply([Add()(third_dense_results_away[:8]), inputs4])
    home_average_team_embedding9 = multiply([Add()(third_dense_results_home[:9]), inputs3])
    away_average_team_embedding9 = multiply([Add()(third_dense_results_away[:9]), inputs4])
    home_average_team_embedding10 = multiply([Add()(third_dense_results_home[:10]), inputs3])
    away_average_team_embedding10 = multiply([Add()(third_dense_results_away[:10]), inputs4])
    home_average_team_embedding11 = multiply([Add()(third_dense_results_home[:11]), inputs3])
    away_average_team_embedding11 = multiply([Add()(third_dense_results_away[:11]), inputs4])
    home_average_team_embedding12 = multiply([Add()(third_dense_results_home[:12]), inputs3])
    away_average_team_embedding12 = multiply([Add()(third_dense_results_away[:12]), inputs4])
    home_average_team_embedding13 = multiply([Add()(third_dense_results_home[:13]), inputs3])
    away_average_team_embedding13 = multiply([Add()(third_dense_results_away[:13]), inputs4])

    home_list = []
    away_list = []
    for k in range(len(home_pts)):
        if k in home_5[0]:
            home_list.append(home_average_team_embedding5[k])
        if k in away_5[0]:
            away_list.append(away_average_team_embedding5[k])
        if k in home_6[0]:
            home_list.append(home_average_team_embedding6[k])
        if k in away_6[0]:
            away_list.append(away_average_team_embedding6[k])
        if k in home_7[0]:
            home_list.append(home_average_team_embedding7[k])
        if k in away_7[0]:
            away_list.append(away_average_team_embedding7[k])
        if k in home_8[0]:
            home_list.append(home_average_team_embedding8[k])
        if k in away_8[0]:
            away_list.append(away_average_team_embedding8[k])
        if k in home_9[0]:
            home_list.append(home_average_team_embedding9[k])
        if k in away_9[0]:
            away_list.append(away_average_team_embedding9[k])
        if k in home_10[0]:
            home_list.append(home_average_team_embedding10[k])
        if k in away_10[0]:
            away_list.append(away_average_team_embedding10[k])
        if k in home_11[0]:
            home_list.append(home_average_team_embedding11[k])
        if k in away_11[0]:
            away_list.append(away_average_team_embedding11[k])
        if k in home_12[0]:
            home_list.append(home_average_team_embedding12[k])
        if k in away_12[0]:
            away_list.append(away_average_team_embedding12[k])
        if k in home_13[0]:
            home_list.append(home_average_team_embedding13[k])
        if k in away_13[0]:
            away_list.append(away_average_team_embedding13[k])

    home_average_team_embedding = tf.stack(home_list)
    away_average_team_embedding = tf.stack(away_list)
    if opt:
        return home_average_team_embedding
    else:
        return away_average_team_embedding


def get_model(input_shape, train_home_list, train_away_list, val_home_list, val_away_list):
    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)
    inputs3 = Input((1,))
    inputs4 = Input((1,))
    is_train = Input((1,))

    players_home = []
    for k in range(input_shape[-1]):
        sliced_tensor = Lambda(lambda x: x[:, :, :, k], output_shape=input_shape[:-1])(inputs1)
        players_home.append(Flatten()((sliced_tensor)))
    players_away = []
    for k in range(input_shape[-1]):
        sliced_tensor = Lambda(lambda x: x[:, :, :, k], output_shape=input_shape[:-1])(inputs2)
        players_away.append(Flatten()((sliced_tensor)))

    first_drop = Dropout(0.5)
    first_dense_layer_home = Dense(500, activation='relu')
    first_dense_layer_away = Dense(500, activation='relu')
    first_dense_results_home = []
    first_dense_results_away = []
    for k in range(input_shape[-1]):
        first_dense_results_home.append(first_drop(first_dense_layer_home(players_home[k])))
        first_dense_results_away.append(first_drop(first_dense_layer_away(players_away[k])))

    second_drop = Dropout(0.5)
    second_dense_layer_home = Dense(300, activation='relu')
    second_dense_layer_away = Dense(300, activation='relu')
    second_dense_results_home = []
    second_dense_results_away = []
    for k in range(input_shape[-1]):
        second_dense_results_home.append(second_drop(second_dense_layer_home(first_dense_results_home[k])))
        second_dense_results_away.append(second_drop(second_dense_layer_away(first_dense_results_away[k])))

    third_drop = Dropout(0.5)
    third_dense_layer_home = Dense(200, activation='relu', name='dense_home3')
    third_dense_layer_away = Dense(200, activation='relu')
    third_dense_results_home = []
    third_dense_results_away = []
    for k in range(input_shape[-1]):
        third_res_home = third_dense_layer_home(second_dense_results_home[k])
        third_res_away = third_dense_layer_away(second_dense_results_away[k])
        third_dense_results_home.append(third_drop(third_res_home))
        third_dense_results_away.append(third_drop(third_res_away))


    is_true = tf.math.count_nonzero([is_train], dtype='bool')
    xx = is_true
    yy = change_results(third_dense_results_home, third_dense_results_away, inputs3, inputs4, 1, train_home_list,
                        train_away_list)
    zz = change_results(third_dense_results_home,third_dense_results_away, inputs3, inputs4, 1, val_home_list,
                        val_away_list)
    home_average_team_embedding = Lambda(lambda x: K.switch(x[0], x[1], x[2]))([xx, yy, zz])
    xx = is_true
    yy = change_results(third_dense_results_home, third_dense_results_away, inputs3, inputs4, 0, train_home_list,
                        train_away_list)
    zz = change_results(third_dense_results_home, third_dense_results_away, inputs3, inputs4, 0, val_home_list,
                        val_away_list)
    away_average_team_embedding = Lambda(lambda x: K.switch(x[0], x[1], x[2]))([xx, yy, zz])

    final_tensor = concatenate([home_average_team_embedding, away_average_team_embedding])

    home_tensor = home_average_team_embedding
    away_tensor = away_average_team_embedding
    aux_output_home = Dense(500, activation='relu')(home_tensor)
    aux_output_home = Dense(1, activation='linear', name='home_pts')(aux_output_home)
    aux_output_away = Dense(500, activation='relu')(away_tensor)
    aux_output_away = Dense(1, activation='linear', name='away_pts')(aux_output_away)

    final_tensor = Dense(250, activation='relu')(final_tensor)  # 10000
    final_tensor = Dropout(0.5)(final_tensor)
    final_tensor = Dense(100, activation='relu')(final_tensor)  # 10000
    final_tensor = Dropout(0.3)(final_tensor)
    final_tensor = Dense(50, activation='relu')(final_tensor)
    final_tensor = Dense(1, activation='linear', name='diff_pts')(final_tensor)

    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4],
                  outputs=[aux_output_home, aux_output_away, final_tensor])

    losses = {"home_pts": "mean_squared_error", "away_pts": "mean_squared_error",
              "diff_pts": "mean_squared_error"}
    metrics = {"diff_pts": "mean_absolute_error"}
    model.compile(optimizer=Adam(lr=1e-4), loss=losses, metrics=metrics)

    return model