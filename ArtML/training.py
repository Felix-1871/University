from imports import *
from prepare_dataset import *
from customise import *

# Load the Dataset for NN training
nn_data = ImageDataset(file_path, active_df, LabEnc, img_size=224)
train_loader, val_loader, test_loader = DataSplitter(nn_data, ratios=[60,25,15], batches=32, shuffle=True, seed=420)
class CNNBackbone(nn.Module):
    def __init__(self, input_height, input_width, conv_channels, kernels, maxpools, lin_channels, dropout, batchnorm):
        """
        Agrs:
            input_height (int):
                image height in pixels
            input_width (int):
                image width in pixels
            conv_channels (list):
                contains the input and output channels for each
                convolutional layer, therefore using a total of
                len(channels)-1 convolutional layers
            kernels (list):
                contains the kernel sizes to be considered per
                convolution. Must have length len(channels)-1
            maxpools (list):
                contains the MaxPool2d kernel sizes to be considered
                per convolution. Must have length len(channels)-1
            lin_channels (list):
                contains the output channels for each linear layer
                following the convolutions, therefore using a total of
                len(lin_channels) linear layers.
                Note that the last element must be equal to the number
                of classes to be determined.
            classes (int):
                number of output features
            dropout (float):
                dropout probability, 0 <= dropout <= 1
            batchnorm (bool):
                boolean parameter to control whether batch normalization
                is applied or not.
        """
        super(CNNBackbone, self).__init__()
        self.num_conv_layers = len(kernels)
        self.batchnorm = batchnorm
        
        seq = []
        for i in range(self.num_conv_layers):
            seq.append(nn.Conv2d(in_channels=conv_channels[i], 
                                 out_channels=conv_channels[i+1],
                                 kernel_size=kernels[i], stride=1, padding=1))
            seq.append(nn.ReLU())
            if self.batchnorm:
                seq.append(nn.BatchNorm2d(num_features=conv_channels[i+1],track_running_stats=False))
            seq.append(nn.MaxPool2d(kernel_size=maxpools[i]))
            
        # Flatten the output of the final convolution layer
        seq.append(nn.Flatten())
        
        convolutions = nn.Sequential(*seq)
        
        # Calculation of first linear layer dimensions
        # We build an empty tensor of appropriate size and let him go through
        # the above sequence, in order to calculate the output's size automatically
        first_lin = convolutions(torch.empty(1,conv_channels[0],input_height,input_width)).size(-1)
        
        self.num_lin_layers = len(lin_channels)
        for i in range(self.num_lin_layers):
            if i == self.num_lin_layers-1:
                seq.append(nn.Linear(lin_channels[i-1], lin_channels[i]))
                break
            elif i == 0:
                seq.append(nn.Linear(first_lin, lin_channels[i]))
            else:
                seq.append(nn.Linear(lin_channels[i-1], lin_channels[i]))
            seq.append(nn.ReLU())
            seq.append(nn.Dropout(dropout))
        seq.append(nn.Softmax(1))
                
        self.fitter = nn.Sequential(*seq)

    def forward(self, x):
        """CNN forward
        Args:
            x (torch.Tensor):
                [B, S, F] Batch size x sequence length x feature size
                padded inputs
        Returns:
            torch.Tensor: [B, O] Batch size x CNN output size cnn outputs
        """
        out = self.fitter(x)
        return out
    
def load_backbone_from_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    
# adapted code from this repository: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='model/checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Validation loss increase spotted. Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def training_loop(model, train_dataloader, optimizer, device="cuda"):
    model.train()
    batch_losses = []
            
    for batch in train_dataloader:
        x_batch, y_batch = batch
                
        # Move to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
        # Clear the previous gradients first
        optimizer.zero_grad()
        
        # forward pass
        yhat = model(x_batch) # No unpacking occurs in CNNs
        
        # loss calculation
        loss = loss_function(yhat, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        batch_losses.append(loss.data.item())
        
    train_loss = np.mean(batch_losses)

    return train_loss


def validation_loop(model, val_dataloader, device="cuda"):
    
    model.eval()
    batch_losses = []
    
    for batch in val_dataloader:
        x_batch, y_batch = batch
                
        # Move to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        yhat = model(x_batch) # No unpacking occurs in CNNs
        
        loss = loss_function(yhat, y_batch)
        
        batch_losses.append(loss.data.item())
        
    val_loss = np.mean(batch_losses)

    return val_loss # Return validation_loss and anything else you need


def train(model, train_dataloader, val_dataloader, optimizer, epochs, device="cuda", patience=-1, verbose_ct=100):

    train_losses = []
    val_losses = []
    print(f"Initiating CNN training.")
    model_path = f'model/CNN.pt'
    checkpoint_path = 'model/checkpoint.pt'
        
    if patience != -1:
        early_stopping = EarlyStopping(patience=patience, verbose=False, path=checkpoint_path)

    for epoch in range(epochs):
        
        # Training loop
        train_loss = training_loop(model, train_dataloader, optimizer, device)    
        train_losses.append(train_loss)

        # Validation loop
        with torch.no_grad():

            val_loss = validation_loop(model, val_dataloader, device)
            val_losses.append(val_loss)

        if patience != -1:
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Patience limit reached. Early stopping and going back to last checkpoint.")
                break

        if epoch % verbose_ct == 0:        
            print(f"[{epoch+1}/{epochs}] Training loss: {train_loss:.4f}\t Validation loss: {val_loss:.4f}.")

    if patience != -1 and early_stopping.early_stop == True:
        load_backbone_from_checkpoint(model,checkpoint_path)        

    torch.save(model.state_dict(), model_path)

    print(f"CNN training finished.\n")
    
    return train_losses, val_losses
    
def evaluate(model, test_dataloader, device="cuda"):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            
            x_batch, y_batch = batch
                
            # Move to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            yhat = model(x_batch) # No unpacking occurs in CNNs
            
            # Calculate the index of the maximum argument
            yhat_idx = torch.argmax(yhat, dim=1)
            
            predictions.append(yhat_idx.cpu().numpy())
            labels.append(y_batch.cpu().numpy())
    
    return predictions, labels  # Return the model predictions

# Small code to plot losses after training
def plot_losses(train_losses,val_losses,title):
    plt.plot(train_losses, label="Training loss", color=mycol)
    plt.plot(val_losses, label="Validation loss", color=mycomplcol)
    plt.legend(loc='best')
    plt.ylabel('Mean Loss')
    plt.xlabel('Epochs')
    plt.title(f"Loss graph during the process of training the CNN.")
    plt.savefig(title, bbox_inches='tight')
    plt.show() 
    end = time.time()
   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_height = nn_data[0][0].shape[1]
input_width = nn_data[0][0].shape[2]
conv_channels = [nn_data[0][0].shape[0],4,16,64,128]
kernels = [3,3,3,3]
maxpools = [2,2,2,2]
lin_channels = [256,128,20]
dropout = 0.25
learning_rate = 0.01
weight_decay = 1e-6
patience = 10
verbose_ct = 1

epochs = 25

model = CNNBackbone(input_height, input_width, conv_channels, kernels, maxpools, lin_channels, dropout, batchnorm=True)
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

# Train the model
t_losses, v_losses = train(model, train_loader, val_loader, optimizer, epochs,
                           device=device, patience=patience, verbose_ct = verbose_ct)

# Plot the loss diagram
plot_losses(t_losses, v_losses, 'results/CNN_Training_Loss.pdf')

# Evaluate the model
predictions, labels = evaluate(model, test_loader, device=device)

y_true = np.concatenate(labels, axis=0)
y_pred = np.concatenate(predictions, axis=0)

# A depiction of the Confusion Matrix
matplotlib.rc_file_defaults() # to remove the sns darkgrid style
cfmatrix = confusion_matrix(y_true, y_pred)
plot_cm(cfmatrix,'results/CNN Confusion Matrix',artists)