import random
from csv import reader

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        data1 = reader(file)
        for row in data1:
            if not row:
                continue
            dataset.append(row)
    return dataset
def string_to_float(data, column):
    for row in data:
        row[column] = float(row[column].strip())
    
def string_to_int(data, column):
    copy = list()
    for row in data:
        copy.append(row[column])
    copy = set(copy)
    t_dat = dict()
    for i,value in enumerate(copy):
        t_dat[value]=i
    for row in data:
        row[column]= t_dat[row[column]]
    return t_dat
    
def block_formation(data, no_blocks):
    block_size=int(len(data)/no_blocks)
    copy=list(data)
    data_split=list()
    for j in range(no_blocks):
        block=list()
        while len(block) < block_size:
            index=random.randrange(len(copy))
            block.append(copy.pop(index))
        data_split.append(block)
    return data_split
    
def predict(row , weights):
    y = weights[0]
    for i in range(len(row) - 1):
        y +=  weights[i+1] * row[i]
    return 1.0 if y >= 0.0 else 0.0   
    
def train_weights(train, weights,  l_rate, epochs):
    for i in range(epochs):
        
        value = 0
        h_x = [0.0 for i in range(len(train[0]))]
        for row in train:
                predicted = predict(row ,weights)
                actual = row[-1]
                value = actual - predicted
                h_x[0] += value
                for i in range(len(h_x)-1):
                    h_x[i+1] += value*row[i]
        for i in range(len(h_x)):
                       h_x[i] = h_x[i]/len(train)
        weights[0]=weights[0]+l_rate*h_x[0]
        for i in range(len(weights)-1):
                       weights[i+1]=weights[i+1]+l_rate*h_x[i+1]
    return weights                

def perceptron(train, test, weights, l_rate, epochs):
    weights= train_weights(train, weights, l_rate, epochs)
    score=0
    for row in test:
        predicted = predict(row, weights)
        actual = row[-1]
        if(actual==predicted):
            score +=1    
    return score
    
def evaluate_algorithm(dataset, algorithm, no_blocks, l_rate, epochs):
    scores = list()
    weights=[0.0 for i in range(len(dataset[0][0]))]
    for block in dataset:
        trainset = list(dataset)
        trainset.remove(block)
        trainset= sum(trainset, [])
        testset=list(block)        
        score = algorithm(trainset, testset, weights, l_rate, epochs)
        score = score/float(len(testset)) * 100
        scores.append(score)
    return scores    
        
random.seed(3)
filename = 'sonar.all-data.csv'
data = load_csv(filename)
#print(data[112])
for i in range(len(data[0])-1):
    string_to_float(data, i)
string_to_int(data, len(data[0])-1)
no_blocks = 5
l_rate = 0.01
epochs = 1000
dataset = block_formation(data, no_blocks)
scores = evaluate_algorithm(dataset, perceptron, no_blocks, l_rate, epochs )
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    





        
