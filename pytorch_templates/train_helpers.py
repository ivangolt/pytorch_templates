import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class ClassifierTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        num_epochs=10,
        cuda: bool = True,
    ):

        self.cuda = cuda
        self.model = model
        self.model = self.model.cuda() if self.cuda else self.model.to("cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs

        self.criterion = criterion
        self.optimizer = optimizer

        # Initialize lists for storing loss and accuracy values
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_losses = []

        self.save_best_model = SaveBestModel()

    def train(self, save_model=True):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch : {epoch+1}/{self.num_epochs}",
            )
            for i, (inputs, labels) in progress_bar:

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                progress_bar.set_postfix(
                    Loss=f"{round(loss.item(),4)}",
                    Accuracy=f"{round((100 * correct / total),3)}",
                )

            # Store and print average loss and accuracy per epoch
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            self.train_losses.append(epoch_loss)
            self.train_accs.append(epoch_acc)

            # Validate the model
            val_acc, val_loss = self.validate()
            self.val_accs.append(val_acc)
            self.val_losses.append(val_loss)

            # saving the model
            if save_model:
                self.save_best_model(
                    current_valid_loss=val_loss,
                    model=self.model,
                    epoch=epoch,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                )

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            running_loss = 0.0
            for inputs, labels in self.val_loader:
                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss = running_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        print(f"Validation Accuracy: {accuracy}% and Loss: {loss}")
        return accuracy, loss


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="blue", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("outputs/accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/loss.png")


class SaveBestModel:

    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        os.makedirs("outputs", exist_ok=True)

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                "outputs/best_model.pth",
            )

            # torch.save(model, "outputs/best_model.pth")