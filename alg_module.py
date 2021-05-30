import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import scipy.io
import os


data_dir = 'F:\\DL bootcamp gits\\projects\\winner\\games_data\\'

rel_keys = ['%_1PT','%_2PT','%_3PT','+/-','AS','A_1PT','A_2PT','A_3PT','BKA_Blocks','BKF_Blocks','DNK','DR_Rebounds','FA_Fouls',
            'M/A_1PT','M_1PT','M_2PT','M_3PT','Min','OR_Rebounds','PF_Fouls','Pts']
rel_keys = ['Pts']
games_df   = pd.read_pickle('games_df_clean')
games_df = games_df.sort_values(by ='month_idx' )

players_df = pd.read_pickle('players_df_clean')
players_array = np.array(players_df[rel_keys])
players_ids   = np.array(players_df['player_id'])
spid = sorted([int(pid) for pid in np.unique(players_ids)])
NP = 13
pid_indexes = {sp:i for i,sp in enumerate(spid)}
nplayers = len(spid)

def make_data():
    all_games = []
    counter = 0
    data_bug_counter = 0
    for irow,game_df in games_df.iterrows():
        counter+=1
        print(counter/len(games_df))
        game_id = game_df['game_id']
        rel_players     = players_array[players_df['game_id']==str(game_id)]
        rel_players_ids = players_ids[players_df['game_id'] == str(game_id)]

        rel_players_is_valid = 1-np.isnan(rel_players)
        rel_players[np.isnan(rel_players)] = 0

        home_players_data_array        = np.zeros((nplayers,len(rel_keys)))
        home_players_is_valid_array    = np.zeros((nplayers, len(rel_keys)))
        for iplay in range(NP):
            pid = game_df['player_home'+str(iplay)]
            if pid==0:
                break

            ix = np.where(rel_players_ids == str(pid))[0]
            if ix.size!=1:# data bug
                data_bug_counter+=1
                print('data bug',data_bug_counter)
                continue
            ix = ix[0]
            home_players_data_array[pid_indexes[pid],:]     = rel_players[ix, :]
            home_players_is_valid_array[pid_indexes[pid],:] = rel_players_is_valid[ix, :]
        away_players_data_array = np.zeros((nplayers, len(rel_keys)))
        away_players_is_valid_array = np.zeros((nplayers, len(rel_keys)))
        for iplay in range(NP):
            pid = game_df['player_away' + str(iplay)]
            if pid == 0:
                break
            if  np.where(rel_players_ids == str(pid))[0].size==0:
                continue
            ix = np.where(rel_players_ids == str(pid))[0]
            if ix.size!=1:# data bug
                data_bug_counter+=1
                print('data bug',data_bug_counter)
                continue
            away_players_data_array[pid_indexes[pid], :] = rel_players[ix, :]
            away_players_is_valid_array[pid_indexes[pid], :] = rel_players_is_valid[ix, :]

        cgame = {}
        cgame['home_data'] = home_players_data_array
        cgame['home_data_is_valid'] = home_players_is_valid_array
        cgame['away_data'] = away_players_data_array
        cgame['away_data_is_valid'] = away_players_is_valid_array
        cgame['result'] =game_df['home_team_points']-game_df['away_team_points']
        with open(data_dir+str(counter)+'.pkl','wb') as fid:
            pickle.dump(cgame,fid)

    with open(data_dir + os.listdir(data_dir)[0], 'rb') as fid:
        min_cgame = pickle.load(fid)
    with open(data_dir + os.listdir(data_dir)[0], 'rb') as fid:
        max_cgame = pickle.load(fid)

    for file in os.listdir(data_dir):
        if not 'winner' in file:
            continue
        with open(data_dir + file, 'rb') as fid:
            cgame = pickle.load(fid)
        for k in cgame.keys():
            max_cgame[k] = np.maximum(cgame[k], max_cgame[k])
            min_cgame[k] = np.minimum(cgame[k], min_cgame[k])

    for file in os.listdir(data_dir):
        if not 'winner' in file:
            continue
        with open(data_dir + file, 'rb') as fid:
            cgame = pickle.load(fid)
        for k in cgame.keys():
            cgame[k] = (cgame[k] - min_cgame[k]) / (1e-3 + max_cgame[k] - min_cgame[k])
        with open(data_dir + file, 'wb') as fid:
            pickle.dump(cgame, fid)

    with open('norm_cgame.pkl', 'wb') as fid:
        pickle.dump({'min_cgame': min_cgame, 'max_cgame': max_cgame}, fid)
#make_data()
vector_size = 200
bsize = 1

with tf.variable_scope('net',use_resource=True,reuse=tf.AUTO_REUSE):

    starting_home_players_vector  = home_players_vector  = tf.get_variable('home_players_vector'  ,dtype='float32',shape = [np.unique(players_ids).size,vector_size],initializer=tf.zeros_initializer(),trainable=False)
    starting_home_players_counter = home_players_counter = tf.get_variable('home_players_counter' ,dtype='float32',shape = [np.unique(players_ids).size,1]          ,initializer=tf.zeros_initializer(),trainable=False)
    starting_away_players_vector  = away_players_vector  = tf.get_variable('away_players_vector'  ,dtype='float32',shape = [np.unique(players_ids).size,vector_size],initializer=tf.zeros_initializer(),trainable=False)
    starting_away_players_counter = away_players_counter = tf.get_variable('players_counter'      ,dtype='float32',shape = [np.unique(players_ids).size,1]          ,initializer=tf.zeros_initializer(),trainable=False)

    # avg_home_player = tf.get_variable('avg_home_player' ,dtype='float32',shape=[np.unique(players_ids).size, vector_size], trainable=True)
    # avg_away_player  = tf.get_variable('avg_away_player',dtype='float32',shape=[np.unique(players_ids).size, vector_size], trainable=True)

    ph_home_players = tf.placeholder('float32',[bsize,np.unique(players_ids).size])
    ph_away_players = tf.placeholder('float32',[bsize,np.unique(players_ids).size])
    ph_diff_home_out= tf.placeholder('float32',[bsize,1])

    mn_home_players_vector = home_players_vector/(home_players_counter+1e-3)
    mn_away_players_vector = away_players_vector/(away_players_counter+1e-3)
    
    home_team_vec    = tf.matmul(ph_home_players,mn_home_players_vector)/(1e-10+tf.reduce_sum(ph_home_players,-1,True))
    away_team_vec    = tf.matmul(ph_away_players,mn_away_players_vector)/(1e-10+tf.reduce_sum(ph_away_players,-1,True))

    tot_v = tf.concat([home_team_vec,away_team_vec],-1)
    layer_out = tot_v
    for nout in [100,50,25]:
        layer_out = tf.layers.dense(layer_out,nout,tf.nn.relu)
    layer_out = tf.layers.dense(layer_out,1)

    loss = tf.losses.mean_squared_error(layer_out,ph_diff_home_out)#?tf.reduce_sum(ph_home_players)*tf.reduce_sum(ph_away_players)*
    trainer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    ph_home_players_data          = tf.placeholder('float32',[bsize,np.unique(players_ids).size,len(rel_keys)])
    ph_home_players_data_is_valid = tf.placeholder('float32',[bsize,np.unique(players_ids).size,len(rel_keys)])

    ph_away_players_data          = tf.placeholder('float32',[bsize,np.unique(players_ids).size,len(rel_keys)])
    ph_away_players_data_is_valid = tf.placeholder('float32',[bsize,np.unique(players_ids).size,len(rel_keys)])

    extra_data = [tf.tile(tf.expand_dims(tot_v, 1), [1, ph_away_players_data.shape[1].value, 1]),tf.tile(tf.expand_dims(ph_diff_home_out, 1), [1, ph_away_players_data.shape[1].value, 1])]


    ww = tf.get_variable('w_home_start',[1,1,len(rel_keys),vector_size])
    bb = tf.get_variable('b_home_start',[1,1,len(rel_keys),1])
    isvalid = tf.expand_dims(ph_home_players_data_is_valid,-1)
    home_layer_players = tf.reduce_sum((tf.expand_dims(ph_home_players_data,-1)*ww+bb)*isvalid,2)/(tf.reduce_sum(isvalid,2)+1e-10)
    home_layer_players = tf.concat([home_layer_players]+extra_data,-1)

    for nout in [vector_size,vector_size,vector_size]:
        home_layer_players = tf.layers.conv1d(home_layer_players, nout,1,activation=tf.nn.relu)
    home_layer_players = tf.layers.conv1d(home_layer_players, nout,1,activation=tf.nn.sigmoid)


    ww = tf.get_variable('w_away_start',[1,1,len(rel_keys),vector_size])
    bb = tf.get_variable('b_away_start',[1,1,len(rel_keys),1])
    isvalid = tf.expand_dims(ph_away_players_data_is_valid,-1)
    away_layer_players = tf.reduce_sum((tf.expand_dims(ph_away_players_data,-1)*ww+bb)*isvalid,2)/(tf.reduce_sum(isvalid,2)+1e-10)

    away_layer_players = tf.concat([away_layer_players]+extra_data,-1)
    for nout in [vector_size,vector_size,vector_size]:
        away_layer_players = tf.layers.conv1d(away_layer_players, nout,1,activation=tf.nn.relu)
    away_layer_players = tf.layers.conv1d(away_layer_players, nout,1,activation=tf.nn.sigmoid)


    @tf.custom_gradient
    def mod_assign_home(value):
        a = tf.get_variable('home_players_vector')
        phi = tf.assign(a, value)

        def grad(dy, variables=[tf.get_variable('home_players_vector')]):
            return dy

        return phi, grad

    @tf.custom_gradient
    def mod_assign_away(value):
        a = tf.get_variable('away_players_vector')
        phi = tf.assign(a, value)

        def grad(dy, variables=[tf.get_variable('away_players_vector')]):
            return dy

        return phi, grad

    as_home_players_vector = mod_assign_home( home_players_vector + tf.transpose(ph_home_players)*tf.squeeze(home_layer_players,0))
    as_away_players_vector = mod_assign_away( away_players_vector + tf.transpose(ph_home_players)*tf.squeeze(away_layer_players,0))

    as_home_players_counter = tf.assign(home_players_counter, home_players_counter + tf.transpose(ph_home_players))
    as_away_players_counter = tf.assign(away_players_counter, away_players_counter + tf.transpose(ph_away_players))
    home_players_vector = tf.get_variable('home_players_vector')
    away_players_vector = tf.get_variable('away_players_vector')


all_losses = []
guesses = []
labels  = []
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
with open('norm_cgame.pkl', 'rb') as fid:
    norm_cg = pickle.load( fid)
min_cgame = norm_cg['min_cgame']
max_cgame = norm_cg['max_cgame']

for epoch in range(500):
    init_vars = [starting_home_players_vector,starting_home_players_counter,starting_away_players_vector,starting_away_players_counter]
    sess.run(tf.initialize_variables(init_vars))
    for counter,file in enumerate(os.listdir(data_dir)[0:1000]):
        with open(data_dir + file, 'rb') as fid:
            cgame = pickle.load(fid)
        feed_dict = {}

        are_playing_home = (cgame['home_data'] != 0).any(-1)
        are_playing_away = (cgame['away_data'] != 0).any(-1)

        feed_dict[ph_home_players]  = [are_playing_home]
        feed_dict[ph_away_players]  = [are_playing_away]
        feed_dict[ph_diff_home_out] = [[(cgame['result']*(max_cgame['result']-min_cgame['result'])+min_cgame['result'])/15]]

        feed_dict[ph_home_players_data]          = [cgame['home_data']]
        feed_dict[ph_home_players_data_is_valid] = [cgame['home_data_is_valid']]

        feed_dict[ph_away_players_data]          = [cgame['away_data']]
        feed_dict[ph_away_players_data_is_valid] = [cgame['away_data_is_valid']]


        # _,ll = sess.run([as_home_players_vector,loss],feed_dict)
        _,as_home_players_vector_out,as_away_players_vector_out,_,_,ll,gg,rr,h_counter,a_counter,h_vec,a_vec = sess.run([trainer,as_home_players_vector,as_away_players_vector,as_home_players_counter,as_away_players_counter,loss,layer_out,ph_diff_home_out,home_players_counter,away_players_counter,home_layer_players,away_players_vector],feed_dict)
        all_losses.append(ll**0.5*15)
        #
        # guesses.append(gg.squeeze()*15)
        # labels.append(rr.squeeze()*15)

        print(epoch,counter,np.mean(all_losses[-100::]))
        #
    scipy.io.savemat('all_losses.mat',{'all_losses':all_losses,'guesses':guesses,'labels':labels,'h_counter':h_counter,'a_counter':a_counter,'h_vec':h_vec,'a_vec':a_vec})
    scipy.io.savemat('all_vecs.mat', {'home_vector': as_home_players_vector_out, 'away_vector': as_away_players_vector_out})