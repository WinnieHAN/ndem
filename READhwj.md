D-EM:
line 127 of run_inner_rnn() in nem_model.py: 
        preds, h_new = self.cell((reshaped_masked_deltas, input_data), h_old) # add input_data as context
        # preds, h_new = self.cell(reshaped_masked_deltas, h_old)
line 394 of build_network()  in  network.py:
        # DEBUG
        cell = NEMOutputDiscriWrapper(cell, out_size, "multi_rnn_cell/cell_0/EMCell_discri")
line of 371 of NEMOutputDiscriWrapper  in network.py:
        dense2 = tf.layers.dense(inputs=dense1, units=784*3, activation=tf.nn.relu)   # *3 because the k:=3

D-RNN-EM:
127 same with the bove 
line 469 of build_network()  in  network.py:
        # DEBUG
        cell = NEMOutputDiscriWrapper(cell, out_size, "multi_rnn_cell/cell_0/EMCell_discri")

add v
