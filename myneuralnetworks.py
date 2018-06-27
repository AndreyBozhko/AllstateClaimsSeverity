import numpy as np
import random

class _GenericNN(object):
    
    
    def __init__(self, inner_layers_shape=[], L2=1.0, batch_size=None, random_seed=None, 
                       learning_rate=1.0, max_iter=None, verbose=0):
        """ initialize parameters of neural network:
        
               inner_layers_shape: list of int, each element is # of units in the inner layer
               L2: L2 regularization parameter
               batch_size: size of batch for stochastic gradient descent
               random_seed: seed value for random shuffling
               step: stochastic gradient descent step
               max_iter: maximum number of iterations over the training dataset
               verbose: if 0, no itermediate steps are printed  
               
        """
        self.inner_layers_shape = inner_layers_shape
        self.L2 = L2
        self.batch_size = batch_size
        self.__random_seed = random_seed
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.__verbose = verbose
        
        self.activation_functions = ["linear"]*(len(inner_layers_shape)) + ["linear"]
        self.__activation_dict = {"linear":  [lambda x: np.array(x), 
                                              lambda x: np.ones(np.array(x).shape)], 
                                  "square":  [lambda x: np.array(x)**2,
                                              lambda x: 2.0*np.array(x)],
                                  "cube":    [lambda x: np.array(x)**3,
                                              lambda x: 3.0*np.array(x)**2],
                                  "sigmoid": [__sigmoid__, __sigmoidGradient__]}
        
        
    def __loss__(self, output, h):
            return np.sum((output-h)**2)
        
    def __reshapeweightvector__(self, weightvector, wdimensions):
        '''reshapes weight vector into list of weight matrices'''
        weights, pos = [], 0
        for i in range(len(wdimensions)):
            weights.append(np.reshape(weightvector[pos:pos+wdimensions[i][0]*wdimensions[i][1]],wdimensions[i]))
            pos = pos + wdimensions[i][0]*wdimensions[i][1]
        return weights[:]
    
    
    def __cost_function__(self, features, output, weightvector, wdimensions, L2):
        '''calculates logistic regression cost function and its gradient'''
        weights = self.__reshapeweightvector__(weightvector, wdimensions)
    
        cost = 0
        alist, zlist = [], []
        grad = [(w-w)[:] for w in weights]
    
        N = features.shape[0]
        a = features[:]
        for i in range(len(wdimensions)):
            a = np.hstack((np.ones((N,1)), a))                                              
            z = np.dot(a, np.transpose(weights[i]))
            alist.append(a[:])
            zlist.append(z[:])
            a = self.__activation_dict[self.activation_functions[i]][0](z)
        h = a[:]
    
        cost = np.sum(1.0/N*self.__loss__(output, h))    # compute cost

        for i in (range(len(wdimensions))):                                          # add regularization
            w_reduced = (weights[i])[:,1:]
            w_reduced = np.ndarray.flatten(w_reduced)
            cost = cost + L2/2.0/N*np.dot(w_reduced, w_reduced)

        for i in range(len(wdimensions)-1,-1,-1):                                   # compute gradients
            if i == len(wdimensions)-1:
                delta = np.array(h - output)
            else:
                delta = np.dot(delta, weights[i+1]) * np.hstack((np.ones((len(zlist[i]),1)),
                                                                   self.__activation_dict[self.activation_functions[i]][1](zlist[i])))
                delta = delta[:,1:]
            grad[i] = grad[i] + 1/float(N)*(np.dot(np.transpose(delta), alist[i]))
        
        for i in range(len(wdimensions)):                                            # add gradient regularization
            grad[i] = grad[i] + L2/float(N)*np.hstack(([[0]]*wdimensions[i][0], weights[i][:,1:]))
        
        grad0 = []                                                                   # flatten the gradient vector
        for el in grad:
            grad0 = grad0 + list(np.ndarray.flatten(el))
        
        return (cost, np.array(grad0))

    
    
    def __batch_gradient_descent__(self, X_train, y_train, weights, dim, 
                               L2, batch_size, random_seed, learning_rate, max_iter, verbose):
        """stochastic batch gradient descent"""
        if batch_size == None:
            batch_size = X_train.shape[0]
        
        opt_weights = weights[:]
        grad = weights[:]-weights[:]
        inds = range(X_train.shape[0])
    
        cost_list, weights_list = [], []
        np.random.seed(random_seed)
    
        iteration = 1
        while iteration <= max_iter:
        
            
            np.random.shuffle(inds)
    
            for j in xrange(X_train.shape[0]/batch_size+1):
                cost_new = 0
                X_train_batch = (X_train[inds])[j*batch_size:(j+1)*batch_size]
                y_train_batch = (y_train[inds])[j*batch_size:(j+1)*batch_size]
                
                if X_train_batch.shape[0] > 0:
                    cs, grad = self.__cost_function__(X_train_batch, y_train_batch, opt_weights, dim, L2)
            
                    cost_new += cs * batch_size / X_train.shape[0]
                    opt_weights -= grad*learning_rate*batch_size/X_train.shape[0]
        
                cost_list += [cost_new]
                weights_list = (weights_list + [opt_weights[:]])[-10:]
        
            printflag = (iteration<=10) or ((iteration<=50) and (iteration%5==0)) or (iteration%10==0)
            if printflag and verbose:
                print "Iteration i =",iteration," Cost function =",cost_new
            iteration += 1
    
        return weights_list[np.argmin(cost_list[-10:])]
    
    
    
    
    def fit(self, X, y, final_dim=1):
        """fits neural network to the training data"""
        X = np.array(X)
        
        #initialize weights
        self.__nn_layers_shape = [X.shape[1]] + self.inner_layers_shape + [final_dim]
        self.__weights_shape = tuple([(self.__nn_layers_shape[i+1],1+self.__nn_layers_shape[i]) 
                                      for i in range(len(self.__nn_layers_shape)-1)])
        self.__weights_length = sum(map(lambda el: el[0]*el[1], self.__weights_shape))
        self.__initial_weights = np.array([random.uniform(0.0,1.0) for i in range(self.__weights_length)])
        
        optimized_weights = self.__batch_gradient_descent__(X, y, self.__initial_weights, self.__weights_shape, 
                                                        self.L2, self.batch_size, self.__random_seed, self.learning_rate, 
                                                        self.max_iter, self.__verbose) ## stochastic batch gradient descent
        self.weights = self.__reshapeweightvector__(optimized_weights, self.__weights_shape)
        
        
        
    def predict(self, X):
        """predicts the class labels"""
        X = np.array(X)
    
        a, N = X[:], X.shape[0]
        for i in range(len(self.__weights_shape)):
            a = np.hstack((np.ones((N,1)), a))
            z = np.dot(a, np.transpose(self.weights[i]))
            a = self.__activation_dict[self.activation_functions[i]][0](z)
    
        prediction = np.reshape(a[:], (N,self.__nn_layers_shape[-1]))
        return prediction
        
        
    def fit_transform(self, X, y):
        """fits neural network to the training data and predicts the labels"""
        self.fit(X, y)
        return self.predict(X)
    
    
    
        
        
        
        
        
        


class NNClassifier(_GenericNN):
    
    
    def __init__(self, inner_layers_shape=[], activation_functions=None, L2=1.0, batch_size=None, random_seed=None, 
                       learning_rate=1.0, max_iter=None, verbose=0):
        
        _GenericNN.__init__(self, inner_layers_shape, L2, batch_size, random_seed, learning_rate, max_iter, verbose)
        self.__activation_dict = {"linear":  [lambda x: np.array(x), 
                                              lambda x: np.ones(np.array(x).shape)], 
                                  "sigmoid": [__sigmoid__, __sigmoidGradient__]}
        if activation_functions == None:
            self.activation_functions = ["sigmoid"]*(len(inner_layers_shape))
        else:
            assert len(activation_functions) == len(inner_layers_shape)
            assert list(np.unique(activation_functions)) == ["linear", "sigmoid"]
            self.activation_functions = activation_functions
        self.activation_functions += ["sigmoid"]
    
    def __loss__(self, output, h):
        return -( np.sum(output*np.log(h)) + np.sum((1.0-output)*np.log(1.0-h)) )
    
    
    def fit(self, X, y):
        """fits neural network to the training data"""
        
        #determine unique labels and perform one-hot encoding of y_train
        self.labels = np.unique(y)
        y_onehot = np.array(map(lambda el: [int(self.labels[i]==el) for i in self.labels], y))
        #if len(self.labels) == 2:
        #    self.labels == self.labels[:1]
        #    y_train_onehot = self.__y_train_onehot[:,:1]
        _GenericNN.fit(self, X, y_onehot, final_dim=self.labels.shape[0])
        print self
    
    
    
    def predict(self, X, probability=False):
        prediction = _GenericNN.predict(self, X)
        if not probability:
            return np.apply_along_axis(lambda el: self.labels[np.argmax(el)], 1, prediction)    #returns label
        else:
            return np.apply_along_axis(lambda el: (np.argmax(el), np.max(el)), 1, prediction)    # returns (label, probability)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class NNRegressor(_GenericNN):
    
    def __init__(self, inner_layers_shape=[], activation_functions=None, L2=1.0, batch_size=None, random_seed=None, 
                       learning_rate=1.0, max_iter=None, verbose=0):
        
        _GenericNN.__init__(self, inner_layers_shape, L2, batch_size, random_seed, learning_rate, max_iter, verbose)
        self.__activation_dict = {"linear":  [lambda x: np.array(x), 
                                              lambda x: np.ones(np.array(x).shape)],
                                  "square":  [lambda x: np.array(x)**2,
                                              lambda x: np.array(x)*2.0],
                                  "sigmoid": [__sigmoid__, __sigmoidGradient__]}
        if activation_functions == None:
            self.activation_functions = ["linear"]*(len(inner_layers_shape))
        else:
            assert len(activation_functions) == len(inner_layers_shape)
            self.activation_functions = activation_functions
        self.activation_functions += ["linear"]
   

    def __loss__(self, output, h):
            return np.sum((output-h)**2)
        
        
    def fit(self, X, y):
        _GenericNN.fit(self, X, np.reshape(y, (y.shape[0],1)), final_dim=1)
        print self
    
    def predict(self, X):
        prediction = _GenericNN.predict(self, X)
        return np.reshape(prediction, (X.shape[0],))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def __sigmoid__(self, x):
    """sigmoid activation function"""
    x = np.array(x)
    return (1.0/(1.0 + np.exp(-x)))
def __sigmoidGradient__(self, x):
    """gradient of the sigmoid activation function"""
    x = np.array(x)
    return (np.exp(-np.abs(x))/(1.0 + np.exp(-np.abs(x)))**2)  
