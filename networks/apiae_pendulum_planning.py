import numpy as np
import tensorflow as tf
import pickle
from .nets import *
from .utils import *

class APIAE(object):
    """ APIAE for posterior inference in the latent space"""
    # Data shape : (Batch(B), Samples(L), Time(K), Dim)
    
    def __init__(self,R,L,K,dt,n_x,n_z,n_u,ur,lr,learn='af',isPlanner=False, useIC=True):   
        self.R = R # the number of improvements
        self.L = L # the number of trajectory sampled
        self.K = K # the number of time steps
        self.dt = dt # time interval
        self.sdt = np.sqrt(dt) # sqrt(dt)
        
        self.n_x = n_x # dimension of x; observation
        self.n_z = n_z # dimension of z; latent space
        self.n_u = n_u # dimension of u; control
        
        self.ur = ur # update rate
        self.lr = lr # learning rate
        self.taus = 1.
        self.tauc = 1.
        self.taui = 1.
        
        self.learn = learn # 'a' - learn apiae, 'i' - learn iwae, 'f' - learn fivo, 'af' - learn apiae w/ resampling
        self.isPlanner=isPlanner # flag whether this network is for the planning or not.
        self.useIC = useIC # flag whether this network uses variational flow or not.
        
        if self.isPlanner:
            self.xseq = tf.placeholder(tf.float32, shape=(None,1,1,self.n_x)) # input objective
        else:
            self.xseq = tf.placeholder(tf.float32, shape=(None,1,self.K,self.n_x)) # input sequence of observations
        self.B = tf.shape(self.xseq)[0] # the number of batch
        
        # Define the networks for dynamics and generative model
        self.dynNet = DynNet(intermediate_units=[16, n_z*n_z+n_z+n_u*n_z], n_z=self.n_z, n_u = self.n_u)
        self.genNet = GenNet(intermediate_units=[128, self.n_x*2], n_z=self.n_z)
        
        # Construct PI-net
        self._create_network()
        
        # Initializing the tensor flow variables and saver
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        
        # Launch the session
        self.sess.run(init)
    

    def _create_network(self):
        self._initialize_network() # Define and initialize variables
        self._create_network_wResample()
        
    def _initialize_network(self):
        # Initialize p0
        if self.isPlanner:
            self.mu0 = tf.placeholder(tf.float32, shape=(1,1,1,self.n_z)) # input sequence of observations
        else:
            self.mu0 = tf.zeros((1,1,1,self.n_z))
        self.Sig0 = 10.*tf.ones((1,1,1,self.n_z)) # initialzie witth arbitrary value
        self.ldet_Sig0 = tf.reduce_sum(tf.log(self.Sig0))
        
        # Call variational flow (q)
        if self.useIC and not self.isPlanner:
            self._infer_control() # Run variational flow
        else:
            self.muhat = self.mu0 # (1,1,1,n_z)
            self.Sighat = self.Sig0 # (B,1,1,n_z)
            self.uffhat = tf.zeros((self.B,1,self.K-1,self.n_u)) # (B,1,K-1,n_u)

    def _infer_control(self):
        self.muhat_layer = tf.layers.Dense(units=self.n_z)
        self.logSighat_layer = tf.layers.Dense(units=self.n_z) # assume diagonal matrix
        self.uffhat_layer = tf.layers.Dense(units=self.n_u)
        
        self.var_layers = [self.muhat_layer, self.logSighat_layer, self.uffhat_layer]
        
        self.cell = tf.contrib.rnn.BasicRNNCell(num_units=8, activation=tf.nn.tanh)
        xseq_reverse = tf.reverse(tf.reshape(self.xseq, (-1,self.K,self.n_x)),axis=[1])
        inputs = tf.unstack(xseq_reverse,axis=1)
        hidden_list, hidden_initial = tf.nn.static_rnn(self.cell, inputs, dtype=tf.float32)
        hidden_states = tf.stack(hidden_list,axis=1) # (B,K,n_h)
        hidden_concat = tf.reverse(tf.concat([hidden_states[:,:-1,:],hidden_states[:,1:,:]],axis=2),axis=[1]) # (B,K-1,2*n_h) 
        
        self.muhat = tf.reshape(self.muhat_layer(hidden_initial), (-1,1,1,self.n_z)) # (B,1,1,n_z)
        logSighat = self.logSighat_layer(hidden_initial) # (B,n_z)
        # self.Sighat = tf.ones_like(self.muhat) * 1e-1
        self.Sighat = tf.reshape(tf.exp(logSighat)+1e-9, (-1,1,1,self.n_z)) # (B,1,1,n_z)
        self.uffhat = tf.reshape(self.uffhat_layer(hidden_concat),(-1,1,self.K-1,self.n_u)) # (B,1,K-1,n_u)

    def plan(self, muhat, Sighat, uffseq):
        # Sampling z0(initial latent state) and dwseq(sequence of dynamic noise)
        z0, dwseq = self.Sampler(muhat, Sighat) 

        # Run dynamics and calculate the cost
        zseq, dwseqnew, alpha, S, bound, _ = self.Simulate(z0, muhat, Sighat, uffseq, dwseq, doResample=False)

        # Reshape alpha
        alpha = tf.expand_dims(tf.expand_dims(alpha,axis=-1),axis=-1) # (B,L,1,1)

        # Update optimal control sequence & initial state dist.
        muhat, Sighat, uffseq = self.Update(zseq, dwseqnew, alpha, muhat, Sighat, uffseq) 

        # Save variables
        self.museq = tf.reduce_sum(alpha*zseq, axis=1, keepdims=True)
        self.zseq = zseq
            
        return muhat, Sighat, uffseq
        
    def _create_network_wResample(self):
        # Initialize
        muhat = self.muhat
        Sighat = self.Sighat
        uffseq = self.uffhat
        
        r = tf.constant(0, dtype='int32')
        
        for r in range(self.R):
            muhat, Sighat, uffseq = self.plan(muhat, Sighat, uffseq)
        
    def Sampler(self, muhat, Sighat):
        # For initial states       
        if self.isPlanner:
            z0 = tf.tile(muhat,(1, self.L, 1,1))
        else:
            epsilon_z = tf.random_normal((self.B,self.L,1,self.n_z), 0., 1., dtype=tf.float32) # (B,L,1,n_z)
            sighat = tf.sqrt(Sighat) # (B,1,1,n_z)
            z0 = muhat + sighat*epsilon_z # (B,L,1,n_z)
        
        # For dynamic noise
        epsilon_u = tf.random_normal((self.B, self.L, self.K-1, self.n_u), 0., 1., dtype=tf.float32) # sample noise from N(0,1)
        
        return z0, epsilon_u*self.sdt

    def Simulate(self, z0, muhat, Sighat, uffseq, dwseq, doResample = True):
        # Load initial states, dynamic noise and control sequence
        dwseq_new = tf.zeros_like(dwseq[:,:,0:1,:]) # (B,L,1,n_u)
        
        zk = z0 # (B,L,1,n_z)
        zseq = z0
        
        # initialize S with initial cost
        ss0 = self.state_cost(self.xseq[:,:,0:1,:], zk) # (B,L,1,1)
        if self.isPlanner:
            S = tf.squeeze(0.0*ss0, axis=[-1,-2]) # (B,L)
        else:
            si0 = self.initial_cost(z0, muhat, Sighat) # (B,L,1,1)
            S = tf.squeeze(si0 + ss0, axis=[-1,-2]) # (B,L)
        log_weight = -S - tf.log(self.L*1.0) # (B,L)
        log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True) # (B,1)
        log_weight = log_weight-log_norm # (B,L)
        
        bound = log_norm # (B,1)

        # Compute optimal control with standardized linear feedback policy
        for k in range(self.K-1):
            # Propagate
            dwk = dwseq[:,:,k:k+1,:] # (B,L,1,n_u)
            uffk = uffseq[:,:,k:k+1,:] # (B,1,1,n_u)
            
            if self.isPlanner:
                zk = self.Propagate(zseq[:,:,k:k+1,:], uffk+1.0*dwk/self.dt)
            else:
                zk = self.Propagate(zseq[:,:,k:k+1,:], uffk+dwk/self.dt)
            
            # Concatenate
            zseq = tf.concat([zseq, zk], axis=2) # (B,L,k+2,n_z)
            dwseq_new = tf.concat([dwseq_new, dwk], axis=2) # (B,L,k+2,n_u)
            
            # Compute control cost
            sck = self.control_cost(uffk, dwk)

            # Update cost
            if self.isPlanner:
                if (k < self.K-2):
                    Sk = tf.squeeze(sck, axis=[-1,-2]) # (B,L)
                else:
                    ssk = self.state_cost(self.xseq, zk) # in fact, xseq is ont sequence for planner. It's just objective image.
                    Sk = tf.squeeze(ssk + sck, axis=[-1,-2])
            else:
                # Compute state cost
                ssk = self.state_cost(self.xseq[:,:,k:k+1,:], zk) # (B,L,1,1)
                Sk = tf.squeeze(ssk + sck, axis=[-1,-2]) # (B,L)
                self.hello = ssk
                
            log_weight = log_weight - Sk # (B,L)
            log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True) # (B,1)
            log_weight = log_weight-log_norm # (B,L)
            
            bound = bound + log_norm
            S = S + Sk # (B,L)
            
            # Resampling
            if doResample:
                ess = 1/tf.reduce_sum(tf.exp(2*log_weight), axis=1, keepdims=True) #(B,1)
                should_resample = tf.cast(ess<0.3*self.L, dtype=tf.int32) #(B,1)
                oind = np.expand_dims(np.arange(self.L),0) # (1,L)

                dist = tf.distributions.Categorical(logits=log_weight)
                rind = tf.stop_gradient(tf.transpose(dist.sample(self.L))) # (B,L)

                new_ind = should_resample*rind + (1-should_resample)*oind #(B,L)
                bat_ind =  tf.tile(tf.expand_dims(tf.range(tf.shape(new_ind)[0]),-1),(1,self.L)) #(B,L)
                gather_ind = tf.stack([bat_ind,new_ind],axis=-1) # (B,L,2) 

                zseq = tf.gather_nd(zseq, gather_ind) # (B,L,k+2,n_z)
                dwseq_new = tf.gather_nd(dwseq_new, gather_ind) # (B,L,k+2,n_u)
                S = tf.gather_nd(S, gather_ind) # (B,L)
                
                should_resample_float = tf.cast(should_resample, dtype=tf.float32)
                log_weight = should_resample_float*tf.ones_like(log_weight)*(-tf.log(self.L*1.0)) + (1.0-should_resample_float)*log_weight
        
        return zseq, dwseq_new[:,:,1:,:], tf.exp(log_weight), S, tf.reduce_sum(bound), tf.reduce_mean(1/tf.reduce_sum(tf.exp(2*log_weight), axis=1))
    
    def Propagate(self, zt, ut):
        """Simulate one-step forward"""
        # Input: zt=(...,n_z), ut=(...,n_u)
        # Output: znext=(...,n_z)

        At, bt, sigmat = self.dynNet.compute_Ab_sigma(zt) # (...,n_z,n_z), (...,n_z,1), (...,n_z,n_u)
        zdott = At@tf.expand_dims(zt, axis=-1) + bt # (...,n_z,1)
        
        znext = zt + tf.squeeze(zdott + sigmat@tf.expand_dims(ut,axis=-1),axis=-1)*self.dt # (...,n_z)
        return znext
    
    def initial_cost(self, z0, muhat, Sighat): 
        """Compute the cost of initial state"""
        q0 = lnorm(z0, muhat, Sighat)
        p0 = 0.5 * tf.reduce_sum(-(z0-self.mu0)**2/self.Sig0 - 0.5*tf.log(2*np.pi), axis=-1, keepdims=True) - 0.5*self.ldet_Sig0
        return self.taui*(q0 - p0) # (B,RL,1,1) 
        
    def control_cost(self, uff, dw):
        """Compute the cost of control input"""
        uTu = tf.reduce_sum(uff**2, axis=-1, keepdims=True) # (B,L,1,1) 
        uTdw = tf.reduce_sum(uff*dw, axis=-1, keepdims=True) # (B,L,1,1) 
        
        return self.tauc*(0.5*uTu*self.dt + uTdw) # (B,L,1,1) 
    
    def state_cost(self,xt_true,zt): # shape of inputs: (..., 1, n_x), (..., 1, n_z)
        """Compute the log-likelihood of observation xt given latent zt"""
        xt_mean, xt_logSig = self.genNet.compute_x(zt)
        xt_Sig = tf.exp(xt_logSig) # + 1e-3
        # xt_Sig = 1.0*tf.ones_like(xt_logSig)
        cost = -lnorm(xt_true, xt_mean, xt_Sig) # (..., 1, 1)
        return self.taus*tf.reduce_sum(cost,axis=[-2,-1],keepdims=True) # (..., 1, 1)
    
    def Update(self, zseq, dwseq, alpha, muhat, Sighat, uffseq):   
        # Compute mean and Cov. of L trajectories
        # muhat_star = tf.reduce_sum(alpha*zseq[:, :, :1, :], axis=1, keepdims=True)  # (B,1,1,n_z)
        # Sighat_star = tf.reduce_sum(alpha*((zseq[:,:,:1,:]-muhat)**2), axis=1, keepdims=True) + 1.0 # (B,1,1,n_z)
        
        if self.isPlanner:
            muhat_star = muhat
            Sighat_star = Sighat
        else:
            muhat_star = (1-self.ur)*muhat + self.ur*tf.reduce_sum(alpha*zseq[:, :, :1, :], axis=1, keepdims=True)  # (B,1,1,n_z)
            Sighat_star = tf.reduce_sum(alpha*((zseq[:,:,:1,:]-muhat)**2), axis=1, keepdims=True) + 1e-1 # (B,1,1,n_z)
        
        # Compute optimal control policy
        uffseq_star = uffseq + self.ur * tf.reduce_sum(alpha*dwseq,axis=1,keepdims=True)/self.dt  # (B,1,K-1,n_u)
        
        return muhat_star, Sighat_star, uffseq_star
        
    def saveWeights(self, filename="weights.pkl"):
        """Save the weights of neural networks"""
        weights = {}
        for i, layer in enumerate(self.dynNet.Layers):
            weights['d_w'+str(i)] = self.sess.run(layer.weights)
        
        for i, layer in enumerate(self.genNet.Layers):
            weights['g_w'+str(i)] = self.sess.run(layer.weights)    
        
        if self.useIC:
            for i, layer in enumerate(self.var_layers):
                weights['v_w'+str(i)] = self.sess.run(layer.weights)    

            weights['v_rnn'] = self.sess.run(self.cell.weights)
        
        filehandler = open(filename,"wb")
        pickle.dump(weights,filehandler)
        filehandler.close()

        print('weight saved in '+filename)

    def restoreWeights(self, filename="weights.pkl"):
        """Load the weights of neural networks"""
        filehandler = open(filename,"rb")
        weights = pickle.load(filehandler)
        filehandler.close()

        for i, layer in enumerate(self.dynNet.Layers):
            for j, w in enumerate(layer.weights):
                self.sess.run(tf.assign(w, weights['d_w'+str(i)][j]))

        for i, layer in enumerate(self.genNet.Layers):
            for j, w in enumerate(layer.weights):
                self.sess.run(tf.assign(w, weights['g_w'+str(i)][j]))  
        
        if self.useIC and not self.isPlanner:
            for i, layer in enumerate(self.var_layers):
                for j, w in enumerate(layer.weights):
                    self.sess.run(tf.assign(w, weights['v_w'+str(i)][j])) 

            for j, w in enumerate(self.cell.weights):        
                self.sess.run(tf.assign(w, weights['v_rnn'][j]))
        
        print('weight restored from '+filename)