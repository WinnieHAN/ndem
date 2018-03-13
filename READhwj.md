EM:
line 127 of run_inner_rnn() in nem_model.py: 
        preds, h_new = self.cell((reshaped_masked_deltas, input_data), h_old) # add input_data as context
        # preds, h_new = self.cell(reshaped_masked_deltas, h_old)
line 394 of build_network()  in  network.py:
        # DEBUG
        cell = NEMOutputDiscriWrapper(cell, out_size, "multi_rnn_cell/cell_0/EMCell_discri")
