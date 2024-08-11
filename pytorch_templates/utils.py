import os
import time
import torch


class ModelCompare:
    def __init__(
        self, model1, model1_info: str, model2, model2_info: str, cuda=False
    ) -> None:
        self.model1 = model1
        self.model1_info = model1_info

        self.model2 = model2
        self.model2_info = model2_info

        self.cuda = cuda
        if cuda:
            self.model1 = self.model1.cuda()
            self.model2 = self.model2.cuda()

    def __calculate_model_size(self, model):
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size

    def __calculate_accuracy(self, model, dataloader):
        model.eval()  # set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # disable gradient calculation
            for data in dataloader:
                images, labels = data
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        overall_accuracy = 100 * correct / total

        return overall_accuracy

    def compare_size(self):
        model1_size = self.__calculate_model_size(self.model1)
        model2_size = self.__calculate_model_size(self.model2)

        print(f"Model {self.model1_info} Size(Mb): {model1_size}")
        print(f"Model {self.model2_info} Size(Mb): {model2_size}")

        if model1_size < model2_size:
            smaller_model = self.model1_info
            percentage_difference = ((model2_size - model1_size) / model2_size) * 100
        elif model2_size < model1_size:
            smaller_model = self.model2_info
            percentage_difference = ((model1_size - model2_size) / model1_size) * 100
        else:
            print("Both models have the same size.")

        print(f"The {smaller_model} is smaller by {percentage_difference:.2f}%.")

    def compare_accuracy(self, dataloder):
        print(
            f"Accuracy of {self.model1_info}: {self.__calculate_accuracy(self.model1,dataloder)}"
        )
        print(
            f"Accuracy of {self.model2_info}: {self.__calculate_accuracy(self.model2,dataloder)}"
        )

    def compare_inference_time(self, N, dataloder):
        total_time_model1 = 0
        total_time_model2 = 0

        # Run inference for model 1
        for _ in range(N):
            start_time = time.time()
            self.__calculate_accuracy(self.model1, dataloder)
            time_taken = time.time() - start_time
            total_time_model1 += time_taken

        # Run inference for model 2
        for _ in range(N):
            start_time = time.time()
            self.__calculate_accuracy(self.model1, dataloder)
            time_taken = time.time() - start_time
            total_time_model2 += time_taken

        # Calculate average inference time
        average_time_model1 = total_time_model1 / N
        average_time_model2 = total_time_model2 / N

        print(
            f"Average inference time of {self.model1_info} over {N} iterations: {average_time_model1}"
        )
        print(
            f"Average inference time of {self.model2_info} over {N} iterations: {average_time_model2}"
        )

        # Compare average inference times
        if average_time_model1 < average_time_model2:
            faster_model = self.model1_info
            percentage_difference = (
                (average_time_model2 - average_time_model1) / average_time_model2
            ) * 100
        elif average_time_model2 < average_time_model1:
            faster_model = self.model2_info
            percentage_difference = (
                (average_time_model1 - average_time_model2) / average_time_model1
            ) * 100
        else:
            print("Both models have the same average inference time.")

        print(f"The {faster_model} is faster by {percentage_difference:.2f}%.")