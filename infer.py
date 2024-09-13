import torch
from torchvision import transforms
from PIL import Image

from data_loader import create_data_loaders, CustomImageFolder
from model import EfficientNetV2

def load_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def infer_images(test_loader, model, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    correct = {classname: 0 for classname in class_names}
    total = {classname: 0 for classname in class_names}
    overall_correct = 0
    overall_total = 0
    results = []

    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for path, pred, label in zip(paths, preds, labels):
                predicted_class = class_names[pred]
                actual_class = class_names[label]

                results.append((path, predicted_class))
                # ファイル名と予測結果を出力
                print(f'File: {path}, Predicted class: {predicted_class}, Actual class: {actual_class}')
                
                # 正解数のカウント
                if pred == label:
                    correct[actual_class] += 1
                    overall_correct += 1
                total[actual_class] += 1
                overall_total += 1

    # 各クラスの精度を計算
    for classname in class_names:
        accuracy = correct[classname] / total[classname] if total[classname] > 0 else 0
        print(f'Accuracy for class {classname}: {accuracy * 100:.2f}%')

    # 全体の精度を計算
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    print(f'Overall Accuracy: {overall_accuracy * 100:.2f}%')

    return results

if __name__ == '__main__':
    data_dir = 'data'
    model_path = 'efficientnet_v2.pth'
    class_names = ['bottle', 'cable', 'capsele']

    train_loader, val_loader, test_loader = create_data_loaders(data_dir)
    num_classes = len(class_names)
    model = load_model(model_path, num_classes)
    infer_images(test_loader, model, class_names)