import torch.nn as nn
import torch
import torch.nn.functional as F
from cm.image_datasets import create_dataset
from cm.script_util import add_dict_to_argparser
import argparse
import os
from torch.utils.data import DataLoader
class LeNet(nn.Module):
    
    def __init__(self,image_size = 28,dataset = 'mnist'):
        super(LeNet, self).__init__()
        self.num_classes = 10
        if dataset =='mnist':
            image_size = 28
            in_channels = 1
        elif dataset =='mnistm':
            image_size = 28
            in_channels = 3
        elif dataset =='svhn':
            image_size = 32
            in_channels = 3
        elif dataset == 'domainNet':
            image_size = 64
            in_channels = 3
            self.num_classes = 345
        else:
            image_size = 64
            in_channels = 3
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        
        )
        if image_size == 32:
            flattened_unit = 1250
        elif image_size == 64:
            flattened_unit = 8450
        elif image_size == 28:
            flattened_unit = 800
        self.classifier = nn.Sequential(
            nn.Linear(flattened_unit,500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500,self.num_classes),
        )
        self.image_size = image_size
        self.in_channels = in_channels
    
    def reshape(self, x):
        if x.shape[2] > self.image_size or x.shape[3] > self.image_size:
            x_reshape = nn.functional.interpolate(x, (self.image_size, self.image_size))
        elif x.shape[2] == self.image_size and x.shape[3] == self.image_size:
            x_reshape = x
        else:
            raise NotImplementedError
        if x.shape[1] == 3 and self.in_channels == 1:
            x_reshape = x_reshape.mean(dim=1)
        return x_reshape
    
    def forward(self, x, mode="logits"): 
        x = self.reshape(x)
        a1=self.feature_extractor(x)
        a1 = torch.flatten(a1,1)
        a2=self.classifier(a1)
        if mode == "logits":
            return a2
        else:
            raise NotImplementedError

def custom_print(fileName,message):
    with open(fileName,'a') as file:
        print(message,file = file)
    print(message)

def train_loop(dataloader,model,loss_fn,optimizer,print_interval,out_file):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct=  0, 0
    for batch,(X,y) in enumerate(dataloader):
        X = X.to('cuda')
        y = y.to('cuda')
        pred = model(X)
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            correct+= (pred.argmax(1) ==y).type(torch.float).sum().item()
            train_loss+=loss
        if batch % print_interval == 0:
            loss,current = loss.item(),(batch+1)*len(X)
            custom_print(out_file,f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct/=size
    train_loss/=num_batches
    return correct, train_loss  
def test_loop(dataloader,model,loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss,correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to('cuda')
            y = y.to('cuda')
            pred = model(X)
            test_loss+=loss_fn(pred,y)
            correct+= (pred.argmax(1) ==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss

def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-2,
        batch_size = 64,
        epochs = 20,
        print_interval = 2,
        image_size = 64,
        class_cond = True,
        log_dir = "/media/minzhe_guo/ckpt/mnist_classifer",
        domain = None,
        resume = False,
        standard_augment = False,
        class_label = 0
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    train_dataset = create_dataset(data_dir=args.data_dir,
        image_size=args.image_size,
        class_cond=args.class_cond,
        train = True,
        domain = args.domain,
        standard_augment = args.standard_augment,
        class_label= args.class_label
    )
    test_dataset = create_dataset(data_dir=args.data_dir,
        image_size=args.image_size,
        class_cond=args.class_cond,
        train = False,
        domain = args.domain,
        standard_augment=args.standard_augment,
        class_label=args.class_label
        
    )

    train_loader= DataLoader(
            train_dataset, batch_size=min(args.batch_size,len(train_dataset)), shuffle=False, num_workers=1, drop_last=True
    )
    test_loader= DataLoader(
            test_dataset, batch_size=min(args.batch_size,len(test_dataset)), shuffle=False, num_workers=1, drop_last=True
    )

    loss_fn = nn.CrossEntropyLoss()
    model = LeNet(image_size=args.image_size,dataset = args.data_dir.split('/')[-1]).to('cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.resume and os.path.exists(args.log_dir) and os.path.isdir(args.log_dir):
        target_ckpt = sorted([filename for filename in os.listdir(args.log_dir)],reverse=True)[0]
        ckpt_dict = torch.load(os.path.join(args.log_dir,target_ckpt))
        model.load_state_dict(ckpt_dict['model_state_dict'])
        start_epoch = ckpt_dict['epoch']
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        print('loading ckpt...')
    else:
        start_epoch = 0 
    os.makedirs(args.log_dir,exist_ok=True)
    best_test_accuracy = 0
    if 'color_digits' in args.data_dir:
        type = 'non_diff_augmented'
    else:
        type = os.path.basename(args.data_dir)
    log_file = os.path.join(args.log_dir,f'{type}_{args.class_label}_log.txt')
    best_epoch = 0
    for t in range(start_epoch,args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        correct,train_loss = train_loop(train_loader, model, loss_fn, optimizer,print_interval=args.print_interval,out_file = log_file)
        custom_print(log_file,f"Training Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        correct,test_loss = test_loop(test_loader, model, loss_fn)
        if correct > best_test_accuracy:
            best_test_accuracy = correct
            best_epoch = t+1
        custom_print(log_file,f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        if (t+1)%50 ==0:
            torch.save({'model_state_dict':model.state_dict(),
                        'epoch':t+1,
                        'optimizer_state_dict':optimizer.state_dict()},
                       os.path.join(args.log_dir,f"model{t+1:03}.pt"))
    custom_print(log_file,f'Best test accuracy is {(best_test_accuracy*100):>0.1f}%, epoch is {best_epoch}')
