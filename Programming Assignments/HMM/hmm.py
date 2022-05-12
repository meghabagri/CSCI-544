from __future__ import print_function
import numpy as np
import json


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])

        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################

        xt_t0 = Osequence[0]
        xt_t0_index = self.obs_dict[xt_t0]

        # Initialize alpha (t=0) i.e. Base Case
        for s in range(S):
            # find the forward message at time t=0 for both the states
            alpha[s][0] = self.pi[s] * self.B[s][xt_t0_index]

        # calculate the forward message for time t=1 to t=6
        for t in range(1, L):
            # Observation at 't'
            xt = Osequence[t]
            xt_index = self.obs_dict[xt]
            for s in range(S):
                cal = 0
                for s_dash in range(S):
                    cal = cal + self.A[s_dash][s] * alpha[s_dash][t - 1]
                alpha[s][t] = self.B[s][xt_index] * cal
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################

        # initialize beta at t = T (L-1)
        for s in range(S):
            beta[s][L-1] = 1

        T = L-1
        for t in range(T-1, -1, -1):
            for s in range(S):
                cal = 0
                for s_dash in range(S):
                    xt_plus1 = Osequence[t + 1]
                    xt_plus1_index = self.obs_dict[xt_plus1]
                    cal = cal + self.A[s][s_dash] * self.B[s_dash][xt_plus1_index] * beta[s_dash][t + 1]
                beta[s][t] = cal
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################

        # select any random 't' from t=0 to T
        t = np.random.randint(0, len(Osequence)-1, 1)[0]
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        prob = 0
        for s in range(len(self.pi)):
            prob += alpha[s][t] * beta[s][t]
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
        (note that this is gamma[i, t-1] instead of gamma[i, t]
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        seq_prob = np.sum(np.multiply(alpha[:, -1], beta[:, -1]))

        post_prob = np.multiply(alpha, beta) / seq_prob

        return post_prob

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])

        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = np.sum(np.multiply(alpha[:, -1], beta[:, -1]))

        T = L-1
        for t in range(T):
            for s in range(S):
                for s_dash in range(S):
                    x_t_plus1 = self.obs_dict[Osequence[t+1]]
                    prob[s, s_dash, t] = alpha[s, t] * self.A[s, s_dash] * self.B[s_dash, x_t_plus1] *\
                                         beta[s_dash, t + 1] / seq_prob
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################

        S = len(self.pi)
        L = len(Osequence)
        b_s_x1 = self.obs_dict[Osequence[0]]
        delta_st = np.zeros([S, L])
        big_delta_st = np.zeros([S, L], dtype=int)

        # Initialize delta for t=0
        delta_st[:, 0] = np.multiply(self.pi, self.B[:, b_s_x1])

        T = L

        for t in range(1, T):
            for s in range(S):
                temp = np.multiply(self.A[:, s], delta_st[:, t - 1])
                xt = self.obs_dict[Osequence[t]]
                delta_st[s, t] = self.B[s, xt] * np.max(temp, axis=0)
                big_delta_st[s, t] = np.argmax(temp)

        get_index = np.argmax(delta_st[:, T - 1])
        path.append(self.find_key(self.state_dict, get_index))
        for t in range(T - 1, 0, -1):
            zt = big_delta_st[get_index, t]
            get_index = zt
            path.append(self.find_key(self.state_dict, zt))
        path.reverse()
        return path

    # DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
