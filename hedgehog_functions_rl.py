"""
Module for reinforcement learning on chess endgame.
"""


import chess
import numpy as np


def initial_position():
    
    """
    Argument:
    
    Returns:
        board -- initial board with only black king, white queen and white king
    """
    
    while True:
        
        board = chess.Board(fen = None)
        
        board.set_piece_at(np.random.randint(64),chess.Piece(6,1))
        board.set_piece_at(np.random.randint(64),chess.Piece(6,0))
        board.set_piece_at(np.random.randint(64),chess.Piece(5,1))
        
        if board.is_valid(): break

    return board


def features(board):
    
    """
    Argument:
        board -- chess board position with only black king, white queen and white king
    
    Returns:
        x -- features, numpy array of dim (18,1)
        x[0,0] -- is black king attacked by white queen
        x[1,0] -- is white queen attackd by black king
        x[2,0] -- is white king "attacked" by white queen
        x[3,0] -- is white queen "attacked" by white king
        x[4,0] -- number of squares attacked by black king
        x[5,0] -- number of squares attacked by white queen
        x[6,0] -- number of squares attacked by white king
        x[7,0] -- distance between black king and white queen
        x[8,0] -- distance between black king and white king
        x[9,0] -- distance between white queen and white king
        x[10,0] -- file difference between black king and white queen
        x[11,0] -- rank difference between black king and white queen
        x[12,0] -- file difference between black king and white king
        x[13,0] -- rank difference between black king and white king
        x[14,0] -- file difference between white queen and white king
        x[15,0] -- rank difference between white queen and white king
        x[16,0] -- turn  
        x[17,0] -- number of legal moves
    """
    
    x = np.zeros((18,1), dtype = np.int16)
    
    # square, file and rank for black king (bk), white queen (wq), white king (wk)
    
    bk = list(board.pieces(6,0))[0]
    wq = list(board.pieces(5,1))[0]
    wk = list(board.pieces(6,1))[0]
    
    bk_f = chess.square_file(bk)
    bk_r = chess.square_rank(bk)
    wq_f = chess.square_file(wq)
    wq_r = chess.square_rank(wq)
    wk_f = chess.square_file(wk)
    wk_r = chess.square_rank(wk)

    # features
    
    x[0,0] = len(board.attackers(1,bk))
    x[1,0] = len(board.attackers(0,wq))
    x[2,0] = len(board.attackers(1,wk))
    x[3,0] = len(board.attackers(1,wq))

    x[4,0] = len(board.attacks(bk))
    x[5,0] = len(board.attacks(wq))
    x[6,0] = len(board.attacks(wk))
    
    x[7,0] = chess.square_distance(bk,wq)
    x[8,0] = chess.square_distance(bk,wk)
    x[9,0] = chess.square_distance(wq,wk)
    
    x[10,0] = bk_f - wq_f
    x[11,0] = bk_r - wq_r
    x[12,0] = bk_f - wk_f
    x[13,0] = bk_r - wk_r
    x[14,0] = wq_f - wk_f
    x[15,0] = wq_r - wk_r
                    
    x[16,0] = board.turn  
    x[17,0] = len(list(board.legal_moves))

    return x


def relu(x):  
    return x * (x > 0)


def drelu(x):
    return 1. * (x > 0)


class Model():
    
    """
    Model with neural network with two hidden layers and an output layer.
    """    
    
    
    def __init__(self,n_input, n_1, n_2, n_output):
        
        """
        Attributes:
            n_input -- size of the input layer
            n_1 -- size of the first hidden layer
            n_2 -- size of the second hidden layer
            n_output -- size of the output layer
            parameters_count -- number of free parameters
            W1 -- weight matrix of shape (n_1, n_input)
            b1 -- bias vector of shape (n_1, 1)
            W2 -- weight matrix of shape (n_2, n_1)
            b2 -- bias vector of shape (n_2, 1)
            W3 -- weight matrix of shape (n_output, n_2)
            b3 -- bias vector of shape (n_output, 1)  
        """ 
        
        self.n_input = n_input
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_output = n_output
        
        self.parameters_count = (n_input+1)*n_1 + (n_1+1)*n_2 + (n_2+1)*n_output
        
        self.W1 = np.random.randn(n_1, n_input) * np.sqrt(2./n_input)
        self.b1 = np.zeros((n_1,1))
        self.W2 = np.random.randn(n_2, n_1) * np.sqrt(2./n_1)
        self.b2 = np.zeros((n_2,1))
        self.W3 = np.random.randn(n_output, n_2) * np.sqrt(2./n_2)
        self.b3 = np.zeros((n_output,1))
        
        
    def adam_init(self, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8):
        
        """
        Attributes:
            beta1 -- parameter of exponential decay, adam optimizer
            beta2 -- parameter of exponential decay, adam optimizer
            epsilon -- parameter of adam optimizer
            t -- step in adam optimizer
            VdW1... -- exponentially weighted averages for adam optimizer
            SdW1... -- exponentially weighted averages for adam optimizer
        """ 
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.t = 0
        
        self.VdW1 = np.zeros_like(self.W1)
        self.Vdb1 = np.zeros_like(self.b1)
        self.VdW2 = np.zeros_like(self.W2)
        self.Vdb2 = np.zeros_like(self.b2)
        self.VdW3 = np.zeros_like(self.W3)
        self.Vdb3 = np.zeros_like(self.b3)
        self.SdW1 = np.zeros_like(self.W1)
        self.Sdb1 = np.zeros_like(self.b1)
        self.SdW2 = np.zeros_like(self.W2)
        self.Sdb2 = np.zeros_like(self.b2)
        self.SdW3 = np.zeros_like(self.W3)
        self.Sdb3 = np.zeros_like(self.b3)
        
    
    def forward_propagation(self, x):
    
        """
        Argument:
            x -- input features
    
        Returns:
            Z1, A1, Z2, A2, Z3 -- outputs of various stages of froward propagation
        """

        Z1 = np.dot(self.W1,x) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(self.W2,A1) + self.b2
        A2 = relu(Z2)
        Z3 = np.dot(self.W3,A2) + self.b3
    
        return Z1, A1, Z2, A2, Z3
    
    
    def backward_propagation(self, x, Z1, A1, Z2, A2, Z3):
    
        """
        Arguments:
            x -- input features
            Z1, A1, Z2, A2 -- outputs of froward propagation

        Returns:
            dW3, db3, dW2, db2, dW1, db1 -- outputs of various stages of backward propagation
        """

        dW3 = A2.T
        db3 = 1
        dZ2 = self.W3.T * drelu(Z2)
        dW2 = np.dot(dZ2,A1.T)
        db2 = dZ2
        dZ1 = np.dot(self.W2.T,dZ2) * drelu(Z1)
        dW1 = np.dot(dZ1,x.T)
        db1 = dZ1

        return dW3, db3, dW2, db2, dW1, db1
    
    
    def adam_update(self, dW1, db1, dW2, db2, dW3, db3):
    
        """
        Arguments: 
            dW1, db1, dW2, db2, dW3, db3 -- gradients for update after single game
        """

        self.VdW1 = self.beta1 * self.VdW1 + (1-self.beta1) * dW1
        self.Vdb1 = self.beta1 * self.Vdb1 + (1-self.beta1) * db1
        self.VdW2 = self.beta1 * self.VdW2 + (1-self.beta1) * dW2
        self.Vdb2 = self.beta1 * self.Vdb2 + (1-self.beta1) * db2
        self.VdW3 = self.beta1 * self.VdW3 + (1-self.beta1) * dW3
        self.Vdb3 = self.beta1 * self.Vdb3 + (1-self.beta1) * db3        

        self.SdW1 = self.beta2 * self.SdW1 + (1-self.beta2) * np.square(dW1)
        self.Sdb1 = self.beta2 * self.Sdb1 + (1-self.beta2) * np.square(db1)
        self.SdW2 = self.beta2 * self.SdW2 + (1-self.beta2) * np.square(dW2)
        self.Sdb2 = self.beta2 * self.Sdb2 + (1-self.beta2) * np.square(db2)
        self.SdW3 = self.beta2 * self.SdW3 + (1-self.beta2) * np.square(dW3)
        self.Sdb3 = self.beta2 * self.Sdb3 + (1-self.beta2) * np.square(db3)
        
        self.t = self.t + 1 

        
    def update_parameters(self, learning_rate = 0.001):
        
        """
        Arguments:
            learning_rate -- learning rate
        """
        
        c1 = 1 - self.beta1 ** self.t
        c2 = (1 - self.beta2 ** self.t) ** 0.5
        
        rate = (learning_rate * c2) / c1
        
        self.W1 = self.W1 + rate * self.VdW1 / (np.sqrt(self.SdW1) + self.epsilon) 
        self.b1 = self.b1 + rate * self.Vdb1 / (np.sqrt(self.Sdb1) + self.epsilon)
        self.W2 = self.W2 + rate * self.VdW2 / (np.sqrt(self.SdW2) + self.epsilon)
        self.b2 = self.b2 + rate * self.Vdb2 / (np.sqrt(self.Sdb2) + self.epsilon)
        self.W3 = self.W3 + rate * self.VdW3 / (np.sqrt(self.SdW3) + self.epsilon)
        self.b3 = self.b3 + rate * self.Vdb3 / (np.sqrt(self.Sdb3) + self.epsilon)


def move_to_play(board, model, features):
    
    """
    Argument:
        board -- chess board position
        model -- the model
        featuers -- function for featuers from board position
                   
    Returns:
        v -- evaluation of the chess board position; values for games that ended are hardcoded
        best_move -- best move in the position
        board_final -- board position after the best move
    """
    
    if board.turn:
        v = 10**6.
        for move in board.legal_moves:
            board.push(move)
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                if result == "1-0":
                    v_temp = -10**6.
                if result == "1/2-1/2":
                    if board.is_stalemate():
                        v_temp = 90000.
                    elif board.can_claim_threefold_repetition():
                        v_temp = 80000.
                    elif board.can_claim_fifty_moves():
                        v_temp = 15.
                    else:
                        v_temp = 15.
            else:
                x = features(board)
                v_temp = model.forward_propagation(x)[4]
            if v_temp < v:
                v = v_temp
                board_final = board.copy()
                best_move = move
            board.pop()
    
    else:
        v = -10**6.
        for move in board.legal_moves:
            board.push(move)
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                if result == "1/2-1/2":
                    if board.is_insufficient_material():
                        v_temp = 10**6.
                    elif board.can_claim_threefold_repetition():
                        v_temp = 80000
                    elif board.can_claim_fifty_moves():
                        v_temp = 15.
                    else:
                        v_temp = 15.
            else:          
                x = features(board)
                v_temp = model.forward_propagation(x)[4]
            if v_temp > v:
                v = v_temp
                board_final = board.copy()
                best_move = move
            board.pop()

    return v, best_move, board_final
    
    
def game_for_update(board, model, features, insufficient_material_value = 30., stalemate_value = 4.):
    
    """
    Arguments:
        board -- input chess board position
        model -- the model
        featuers -- function for featuers from board
        insufficient_material_value -- value assigned to board position with insufficient material
        stalemate_value -- value assigned to stalemate
    
    Returns:
        board -- final position of the game
        values_array -- evaluations of the chess board positions during the game
        dW1_array, db1_array, dW2_array, db2_array, dW3_array, db3_array -- gradients of evaluations 
        of the chess board positions during the game
    """
    
    values_list = []
    dW1_list = []
    db1_list = []
    dW2_list = []
    db2_list = []
    dW3_list = []
    db3_list = []
   
    while not board.is_game_over(claim_draw=True):
        
        x = features(board)
        Z1, A1, Z2, A2, Z3 = model.forward_propagation(x)        
        dW3, db3, dW2, db2, dW1, db1 = model.backward_propagation(x, Z1, A1, Z2, A2, Z3)
        
        values_list.append(np.asscalar(Z3))
        dW1_list.append(dW1)
        db1_list.append(db1)
        dW2_list.append(dW2)
        db2_list.append(db2)
        dW3_list.append(dW3)
        db3_list.append(db3)
        
        value, best_move, board_final = move_to_play(board, model, features)
            
        board.push(best_move)
        
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        if result == "1-0":
            values_list.append(0.)
        if result == "1/2-1/2":
            if board.is_insufficient_material():
                values_list.append(insufficient_material_value)
            elif board.is_stalemate():
                values_list.append(stalemate_value)
            elif board.can_claim_threefold_repetition():
                x = features(board)
                Z1, A1, Z2, A2, Z3 = model.forward_propagation(x)
                values_list.append(np.asscalar(Z3))
            elif board.can_claim_fifty_moves():
                x = features(board)
                Z1, A1, Z2, A2, Z3 = model.forward_propagation(x)
                values_list.append(np.asscalar(Z3))
            else:
                x = features(board)
                Z1, A1, Z2, A2, Z3 = model.forward_propagation(x)
                values_list.append(np.asscalar(Z3))
            
    values_array = np.array(values_list)
    dW1_array = np.array(dW1_list)
    db1_array = np.array(db1_list)
    dW2_array = np.array(dW2_list)
    db2_array = np.array(db2_list)
    dW3_array = np.array(dW3_list)
    db3_array = np.array(db3_list)
    
    return board, values_array, dW1_array, db1_array, dW2_array, db2_array, dW3_array, db3_array
   
    
def update_sums(values_array, dW1_array, db1_array, dW2_array, db2_array, dW3_array, db3_array, lam = 0.5):
    
    """
    Arguments:
        values_array, dW1_array, db1_array, dW2_array, db2_array, dW3_array, db3_array -- input from the game,
        evaluations of the chess board positions during the game and gradients of evaluations 
        of the chess board positions during the game
        lam -- parameter from TD(lambda)
    
    Returns:
        dW1, db1, dW2, db2, dW3, db3 -- gradients for update after single game for temporal difference reinforcement learning
    """
        
    n = len(values_array)
    
    d = 1 + values_array[1:n] - values_array[0:n-1]
    
    dW1 = np.zeros_like(dW1_array[0])
    db1 = np.zeros_like(db1_array[0])
    dW2 = np.zeros_like(dW2_array[0])
    db2 = np.zeros_like(db2_array[0])
    dW3 = np.zeros_like(dW3_array[0])
    db3 = np.zeros_like(db3_array[0])
    
    e_dW1 = np.zeros_like(dW1_array[0])
    e_db1 = np.zeros_like(db1_array[0])
    e_dW2 = np.zeros_like(dW2_array[0])
    e_db2 = np.zeros_like(db2_array[0])
    e_dW3 = np.zeros_like(dW3_array[0])
    e_db3 = np.zeros_like(db3_array[0])
    
    for i in range(0,n-1):
        e_dW1 = lam * e_dW1 + dW1_array[i]
        e_db1 = lam * e_db1 + db1_array[i]
        e_dW2 = lam * e_dW2 + dW2_array[i]
        e_db2 = lam * e_db2 + db2_array[i]
        e_dW3 = lam * e_dW3 + dW3_array[i]
        e_db3 = lam * e_db3 + db3_array[i]
        
        dW1 = dW1 + e_dW1 * d[i]
        db1 = db1 + e_db1 * d[i]
        dW2 = dW2 + e_dW2 * d[i]
        db2 = db2 + e_db2 * d[i]
        dW3 = dW3 + e_dW3 * d[i]
        db3 = db3 + e_db3 * d[i]
    
    return dW1, db1, dW2, db2, dW3, db3
     
