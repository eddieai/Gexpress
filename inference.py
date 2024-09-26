# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from torch.utils.data import DataLoader
cuda = torch.cuda.is_available()
from networks import ClassificationNet


n_classes = 25
n_channels = 42
n_frames = 100
dropout_probability = 1

model = ClassificationNet(n_channels=n_channels, n_frames=n_frames, n_classes=n_classes, dropout_probability=dropout_probability)
if cuda:
    model.cuda()

model.load_state_dict(torch.load('softmax_best.pt', map_location=torch.device('cpu')))
model.eval()


def report():
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_loader):
            if cuda:
                inputs, classes = inputs.cuda(), classes.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    class_report = classification_report(lbllist.numpy(), predlist.numpy())
    print(conf_mat)
    print('\n------------------\n', class_report)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print('\n------------------\n', class_accuracy)


def inference(inputs):
    inputs = torch.tensor(inputs).float()
    with torch.no_grad():
        if cuda:
            inputs = inputs.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    return preds.cpu().numpy().squeeze()


if __name__ == "__main__":
    from datasets import Jester
    from sklearn.metrics import confusion_matrix, classification_report

    dataset_name = 'jester_42'
    batch_size = 128

    dataset = Jester('data/', dataset_name=dataset_name, train=False)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    test_loader = DataLoader(dataset, batch_size=batch_size, **kwargs)

    report()
