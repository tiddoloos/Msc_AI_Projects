from _context import vugrad
import numpy as np
from argparse import ArgumentParser
import vugrad as vg
import matplotlib.pyplot as plt


parser = ArgumentParser()

parser.add_argument('-D', '--dataset',
                dest='data',
                help='Which dataset to use. [synth, mnist]',
                default='synth', type=str)

parser.add_argument('-b', '--batch-size',
                dest='batch_size',
                help='The batch size (how many instances to use for a single forward/backward pass).',
                default=128, type=int)

parser.add_argument('-e', '--epochs',
                dest='epochs',
                help='The number of epochs (complete passes over the complete training data).',
                default=20, type=int)

parser.add_argument('-l', '--learning-rate',
                dest='lr',
                help='The learning rate. That is, a scalar that determines the size of the steps taken by the '
                     'gradient descent algorithm. 0.1 works well for synth, 0.0001 works well for MNIST.',
                default=0.01, type=float)

args = parser.parse_args()

# Load data
if args.data == 'synth':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_synth()
elif args.data == 'mnist':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_mnist(final=False, flatten=True)
    #normalize
    xtrain = xtrain/255
    xval = xval/255
else:
    raise Exception(f'Dataset {args.data} not recognized.')

print(f'## loaded data:')
print(f'         number of instances: {xtrain.shape[0]} in training, {xval.shape[0]} in validation')
print(f' training class distribution: {np.bincount(ytrain)}')
print(f'     val. class distribution: {np.bincount(yval)}')

num_instances, num_features = xtrain.shape


class MLP(vg.Module):
    def __init__(self, input_size, output_size, hidden_mult=4, rel = False):
        self.rel = rel
        super().__init__()
        hidden_size = hidden_mult * input_size
        self.layer1 = vg.Linear(input_size, hidden_size)
        self.layer2 = vg.Linear(hidden_size, output_size)

    def forward(self, input):
        assert len(input.size()) == 2
        # first layer
        hidden = self.layer1(input)
        # non-linearity
        if self.rel == True:
            hidden = vg.relu(hidden)
        else:
            hidden = vg.sigmoid(hidden)
        # second layer
        output = self.layer2(hidden)

        # softmax activation
        output = vg.logsoftmax(output)
        return output

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()


def experiment(num_features, num_classes, rel=False):
    epoch_list = []
    acc_list = []
    if rel:
        mlp = MLP(input_size=num_features, output_size=num_classes, hidden_mult=4, rel=True)
    else:
        mlp = MLP(input_size=num_features, output_size=num_classes, hidden_mult=4, rel=False)
    
    n, m = xtrain.shape
    b = args.batch_size

    print('\n## Starting training')
    for epoch in range(args.epochs):
        print(f'epoch {epoch:03}')
        ## Compute validation accuracy
        o = mlp(vg.TensorNode(xval))
        oval = o.value
        predictions = np.argmax(oval, axis=1)
        num_correct = (predictions == yval).sum()
        acc = num_correct / yval.shape[0]
        epoch_list.append(epoch)
        acc_list.append(acc)

        o.clear() # gc the computation graph
        print(f'       accuracy: {acc:.4}')

        cl = 0.0 # running sum of the training loss
        # batch loop
        for fr in range(0, n, b):

            # The end index of the batch
            to = min(fr + b, n)

            # Slice out the batch and its corresponding target values
            batch, targets = xtrain[fr:to, :], ytrain[fr:to]

            # Wrap the inputs in a Node
            batch = vg.TensorNode(value=batch)
            outputs = mlp(batch)
            loss = vg.logceloss(outputs, targets)
            cl += loss.value
            loss.backward()

            # Apply gradient descent
            for parm in mlp.parameters():
                parm.value -= args.lr * parm.grad
            loss.zero_grad()
            loss.clear()
        print(f'   running loss: {cl/n:.4}')
    return epoch_list, acc_list


epoch_list, acc_list2 = experiment(num_features, num_classes, rel=True)
epoch_list, acc_list = experiment(num_features, num_classes, rel=False)

#plot te data
y1 = acc_list
x1 = epoch_list
plt.plot(x1, y1, label = "Sigmoid")

y2 = acc_list2
x2 = epoch_list

plt.plot(x2, y2, label = "ReLu")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy of the trained network with Sigmoid and Relu')
plt.xticks(np.arange(0, 21, 1.0))
plt.legend()
plt.savefig('experiments/figures/Q9_acc')
plt.show()