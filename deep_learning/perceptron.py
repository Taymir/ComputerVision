import numpy as np

class Perceptron:
    def __init__(self, N, alpha = 0.1):
        #init wight matrix and store learning rate
        self.W = np.random.randn(N+1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs = 10):
        #insert column of 1's as the last entry in the
        #feature matrix - it allows us to treat the bias
        #as trainable parameter within W matrix
        X = np.c_[X, np.ones((X.shape[0]))]


        #training
        for epoch in np.arange(0, epochs):
            #loop over each ind. data point
            for(x, target) in zip(X, y):
                #take the dot product between the input features
                #and the W matrix, then pass this value
                #though the step function to optain the prediction
                p = self.step(np.dot(x, self.W))

                #only perform a weight update if
                #out prediction doesn't match target
                if p != target:
                    #determine error
                    error = p-target

                    #update W matrix
                    self.W += -self.alpha * error * x

        print(self.W)

    def predict(self, X, addBias = True):
        X = np.atleast_2d(X)

        if addBias:
            #insert s column of 1's as the last entry in the feature
            #matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]

        #take the dot product between the input features and the
        #W matrix, then pass the value through the step function
        return self.step(np.dot(X, self.W))

def test(X, y):
    print("Training")
    p = Perceptron(X.shape[1], alpha = 0.1)
    p.fit(X,y,epochs=20)

    print ("Testing")

    for(x, target) in zip(X,y):
        pred = p.predict(x)
        print("data={}, ground-truth={}, pred={}".format(x, target[0], pred))

if __name__ == "__main__":
    #OR
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [1]])
    #AND
    #X = np.array([[0,0], [0,1], [1,0], [1,1]])
    #y = np.array([[0], [0], [0], [1]])
    #XOR
    #X = np.array([[0,0], [0,1], [1,0], [1,1]])
    #y = np.array([[0], [1], [1], [0]])

    test(X, y)
