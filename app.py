# Загрузка модели
class ResNetModel(torch.nn.Module):
    def init(self, num_classes):
        super(ResNetModel, self).init()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Загрузка весов модели
model_path = "3cls_best_model.pt"
num_classes = 3model = ResNetModel(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Трансформация изображения
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
transform = transform.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Lambda(lambdaimg: img[:3]), transforms.Normalize(mean=MEAN, std=STD)])

# Функция для педсказания класса и вероятностей
def predict(image):
    image = transform(image).unsqueeze(0)
    # Добавление batch dimension
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted = torch.max(outputs, 1)
        return predicted.item(), probabilities.squeeze().cpu().numpy()
    
    
