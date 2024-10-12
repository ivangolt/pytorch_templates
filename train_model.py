import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# Путь к папке с изображениями
image_dir = "train_data/labeled_data"

# Чтение таблицы с дефектами
df = pd.read_excel("train_data/defects.xlsx")


# Функция для проверки существования файлов изображений
def file_exists(row):
    img_path = os.path.join(image_dir, row["filename"])
    return os.path.exists(img_path)


# Удаляем записи с отсутствующими файлами изображений
df = df[df.apply(file_exists, axis=1)].reset_index(drop=True)

# Создаем отображение классов в целые числа
classes = df["main_class"].unique()
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
df["label"] = df["main_class"].map(class_to_idx)

# Разделение данных на тренировочные и тестовые после фильтрации
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Класс для кастомного датасета
class DefectDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        # Открываем изображение
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Получаем метку и конвертируем ее в тензор
        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label


# Трансформации для предобработки изображений
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Создаем датасеты и загрузчики данных
train_dataset = DefectDataset(train_df, image_dir, transform=transform)
test_dataset = DefectDataset(test_df, image_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# # Function to display a batch of images
# def show_augmented_images(data_loader, num_images=10):
#     # Get a batch of images and labels
#     images, labels = next(iter(data_loader))

#     # Set up the figure
#     plt.figure(figsize=(15, 8))

#     for i in range(num_images):
#         plt.subplot(2, 5, i + 1)  # Create a grid of subplots (2 rows, 5 columns)
#         plt.imshow(
#             images[i].permute(1, 2, 0).numpy()
#         )  # Convert from Tensor to NumPy array
#         plt.title(f"Label: {labels[i].item()}")  # Show the label
#         plt.axis("off")  # Hide axes

#     plt.tight_layout()
#     plt.show()


# # Call the function to visualize augmented images
# show_augmented_images(train_loader)


# Модификация модели с использованием Dropout
class ModifiedResNet(nn.Module):
    def __init__(self, original_model, num_classes):
        super(ModifiedResNet, self).__init__()
        # Удаляем последний слой fully connected из ResNet и сохраняем остальную часть модели
        self.resnet = nn.Sequential(*list(original_model.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # Адаптивное среднее для уменьшения размерности
        self.dropout = nn.Dropout(0.5)  # Вероятность dropout 50%
        self.fc = nn.Linear(
            2048, num_classes
        )  # Полносвязный слой под наши классы дефектов

    def forward(self, x):
        x = self.resnet(x)  # Извлекаем промежуточные признаки из ResNet
        x = self.adaptive_pool(x)  # Применяем адаптивное среднее
        x = torch.flatten(x, 1)  # Приводим к размеру (batch_size, features)
        x = self.dropout(x)  # Применяем Dropout
        x = self.fc(x)  # Применяем полносвязный слой
        return x


# Теперь создайте модель
num_classes = len(classes)  # Количество уникальных классов дефектов
model = ModifiedResNet(
    models.resnet50(weights=models.ResNet50_Weights.DEFAULT), num_classes
)

# Замораживаем параметры модели, чтобы обучать только последний слой
for param in model.parameters():
    param.requires_grad = False

# Меняем последний fully connected слой под наши классы дефектов
num_classes = len(classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Переносим модель на GPU, если оно доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Определение функции потерь и оптимизатора с L2 регуляризацией
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-5)


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)  # Получаем предсказанный класс
    corrects = torch.sum(preds == labels)  # Сравниваем с реальными метками
    return corrects.item() / len(labels)  # Возвращаем долю правильных предсказаний


# Обновленная функция для обучения модели с вычислением точности
def train_model(
    model, train_loader, test_loader, criterion, optimizer, num_epochs=50, patience=5
):
    model.train()

    best_val_loss = float("inf")  # Изначально ставим наилучший loss бесконечным
    patience_counter = 0  # Счетчик для терпения
    train_losses, test_losses = [], []  # для хранения значений функции потерь
    train_accuracies, test_accuracies = [], []  # для хранения точности

    for epoch in range(num_epochs):
        model.train()  # Важно вернуть модель в режим обучения
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predictions += calculate_accuracy(outputs, labels) * len(labels)
            total_predictions += len(labels)

        # Вычисляем средний loss и точность для тренировочных данных
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Оценка на тестовом наборе данных
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

        # Early Stopping: проверяем, улучшилась ли модель
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            patience_counter = 0  # Сбрасываем счетчик терпения
            # Сохраняем наилучшую модель
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model improved, saving model at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"No improvement, patience counter: {patience_counter}/{patience}")

        # Если модель не улучшалась в течение "patience" эпох, завершаем обучение
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Построение графиков
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)


# Функция для оценки модели на тестовом наборе
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct_predictions += calculate_accuracy(outputs, labels) * len(labels)
            total_predictions += len(labels)

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct_predictions / total_predictions
    return test_loss, test_accuracy


# Функция для построения графиков
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # График функции потерь
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.show()


# Сохранение модели
def save_model(model, optimizer, filename="trained_model.pth"):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )
    print(f"Model saved to {filename}")


def load_model(model, optimizer, filename="trained_model.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()  # Переводим модель в режим оценки
    print(f"Model loaded from {filename}")  #


train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50)
save_model(model, optimizer)  # Сохраните модель после обучения
