
def model():
    features = [1,2,3] # x
    labels = [2,4,5] # y_ture
    b = 0
    w = 1
    EPOCHS = 10
    learning_rate = 0.01
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i in range(0,len(features)):
            y_pred = LinearModel(w, features[i], b)
            loss = MeanSquaredError(labels[i], y_pred)
            running_loss+= loss
            gradien_w = ChainRuleWeight(features[i], labels[i], y_pred)
            gradien_b = ChainRuleBias(labels[i], y_pred)
            new_w = StochasticGradientDescentWeight(w, learning_rate, gradien_w)
            new_b = StochasticGradientDescentBias(b, learning_rate, gradien_b)
            w = new_w
            b = new_b
            print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f}')
            
        print(f'End of Epoch {epoch+1}: w = {w:.4f}, b = {b:.4f}')
        
    
    return 0
def LinearModel(w,x,b):
    """
    Fungsi untuk model regresi linear sederhana.
    """
    # Model regresi linear: y_pred = w * x + 1
    return w * x + b
    
def MeanSquaredError(y_true, y_pred):
    """
    Fungsi untuk menghitung Mean Squared Error (MSE)
    MSE = 1/n * Σ(y_true - y_pred)²
    """
    return (y_true - y_pred) ** 2

def ChainRuleWeight(x, y_true, y_pred):
    """
    Fungsi untuk menghitung gradien terhadap w (weight) dengan chain rule
    ∂loss/∂b = -2 * (y_true - y_pred) * x 
    """
    return -2 * (y_true - y_pred) * x

def ChainRuleBias( y_true, y_pred):
    """
    Fungsi untuk menghitung gradien terhadap b (bias) dengan chain rule
    ∂loss/∂b = -2 * (y_true - y_pred)
    """
    return -2 * (y_true - y_pred)

def StochasticGradientDescentWeight(w, learning_rate, gradien_w):
    """Fungsi untuk menghitung SGD yang akan update w
    w_new = w - learning_rate * ∂loss/∂w
    Args:
        learning_rate (_type_): _description_
    """
    return w - learning_rate * gradien_w
def StochasticGradientDescentBias(b,learning_rate, gradien_b):
    """Fungsi untuk menghitung SGD yang akan update w
    b_new = b - learning_rate * ∂loss/∂b
    Args:
        learning_rate (_type_): _description_
    """
    return b - learning_rate * gradien_b

if __name__ == "__main__":
    model()