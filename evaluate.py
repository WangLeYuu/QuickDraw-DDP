from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data
import onnxruntime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm
from getdata import mean, std, class_names
from option import get_args
opt = get_args()
device = 'cuda:1'

"""Predicting a single image"""
def evaluate_image_single(img_path, transform_test, model, class_names, top_k):
    
    image = Image.open(img_path).convert('RGB')
    img = transform_test(image).to(device)
    img = img.unsqueeze_(0)
    out = model(img)
    pred_softmax = F.softmax(out, dim=1)
    top_n, top_n_indices = torch.topk(pred_softmax, top_k)
    
    confs = top_n[0].cpu().detach().numpy().tolist()
    class_names_top = [class_names[i] for i in top_n_indices[0]]
    
    for i in range(top_k):
        print(f'Pre: {class_names_top[i]}   Conf: {confs[i]:.4f}')
    
    confs_max = confs[0]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title(f'Pre: {class_names_top[0]}   Conf: {confs_max:.4f}')
    plt.imshow(image)
    
    sorted_pairs = sorted(zip(class_names_top, confs), key=lambda x: x[1], reverse=True)
    sorted_class_names_top, sorted_confs = zip(*sorted_pairs)
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(sorted_class_names_top, sorted_confs, color='lightcoral')
    plt.xlabel('Class Names')
    plt.ylabel('Confidence')
    plt.title('Top 5 Predictions (Descending Order)')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    for bar, conf in zip(bars, sorted_confs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{conf:.4f}', ha='center', va='bottom')
    plt.savefig('predict_image_with_bars.jpg')


"""Predicting folder images"""
def evaluate_image_dir(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    correct_top1, correct_top3, correct_top5, total = torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            correct_top1  += (outputs.argmax(1) == labels).type(torch.float).sum()
            _, predicted_top3 = torch.topk(outputs, 3, dim=1)
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            correct_top3 += (predicted_top3[:, :3] == labels.unsqueeze(1).expand_as(predicted_top3)).sum().item()
            correct_top5 += (predicted_top5[:, :5] == labels.unsqueeze(1).expand_as(predicted_top5)).sum().item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    
    top1 = correct_top1 / total
    top3 = correct_top3 / total
    top5 = correct_top5 / total
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    
    accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
    precision = precision_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
    recall = recall_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
    f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
    
    cm = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy())
    report = classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), target_names=class_names)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(report)

    plt.figure(figsize=(100, 100))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
    plt.xticks(rotation=90) 
    plt.yticks(rotation=0)  
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.jpg')

"""Using .onnx model to predict images"""
def evaluate_onnx_model(img_path, data_transform, onnx_model_path, class_names, top_k=5):
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    img_pil = Image.open(img_path).convert('RGB')
    input_img = data_transform(img_pil)
    input_tensor = input_img.unsqueeze(0).numpy()
    ort_inputs = {'input': input_tensor}
    out = ort_session.run(['output'], ort_inputs)[0]
    
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    prob_dist = softmax(out)
    result_dict = {label: float(prob_dist[0][i]) for i, label in enumerate(class_names)}
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))

    for key, value in list(result_dict.items())[:top_k]:
        print(f'Pre: {key}   Conf: {value:.4f}')

    confs_max = list(result_dict.values())[0]
    class_names_top = list(result_dict.keys())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title(f'Pre: {class_names_top[0]}   Conf: {confs_max:.4f}')
    plt.imshow(img_pil)

    plt.subplot(1, 2, 2)
    bars = plt.bar(class_names_top[:top_k], list(result_dict.values())[:top_k], color='lightcoral')
    plt.xlabel('Class Names')
    plt.ylabel('Confidence')
    plt.title('Top 5 Predictions (Descending Order)')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    for bar, conf in zip(bars, list(result_dict.values())[:top_k]):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{conf:.4f}', ha='center', va='bottom')
    plt.savefig('predict_image_with_bars.jpg')



if __name__ == '__main__':
    data_transform = transforms.Compose([transforms.Resize((opt.loadsize, opt.loadsize)), transforms.ToTensor(),transforms.Normalize(mean, std)])
    image_datasets = ImageFolder(opt.dataset_test, data_transform)
    dataloaders = DataLoader(image_datasets, batch_size=512, shuffle=True)
    
    ptl_model_path = opt.checkpoints + 'model.ptl'
    pth_model_path = opt.checkpoints + 'model.pth'
    onnx_model_path = opt.checkpoints + 'model.onnx'
    
    # ptl_model = torch.jit.load(ptl_model_path).to(device)
    pth_model = torch.load(pth_model_path).to(device)
    
    # evaluate_image_single(opt.test_img_path, data_transform, pth_model, class_names, top_k=5)     # Predicting a single image
    evaluate_image_dir(pth_model, dataloaders, class_names)     # Predicting folder images
    # evaluate_onnx_model(opt.test_img_path, data_transform, onnx_model_path, class_names, top_k=5)   # Predicting a single image
