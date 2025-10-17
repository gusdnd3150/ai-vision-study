import traceback
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

class StaticClassifierTrainer:


    # 정적 모델 전용 (이미지 등)
    def __init__(self,model_name, model_dir, train_dir, val_dir,num_classes=2, batch_size=32, lr=0.001, epochs=5):
        self.model_name = model_name
        self.model_dir = model_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.transform = self.set_transform()


        # trainer = DogClassifierTrainer(train_dir="dataset/train",
        #                        val_dir="dataset/val",
        #                        batch_size=32,
        #                        lr=0.001,
        #                        epochs=5)
        # trainer.load_study_data()     # 데이터 로드 + 정규화
        # trainer.build_model()   # 모델 구성
        # trainer.train()         # 학습
        # trainer.evaluate()      # 검증
        # trainer.save_model()    # 모델 저장
        # # 단일 이미지 테스트
        # label_idx = trainer.predict_image("test_dog.jpg")
        # print("🐶 강아지" if label_idx==1 else "😺 다른 동물")

    def loadModel(self):
        self.build_model(self.model_dir)

    def run_train_process(self):
        try:
            self.load_study_data()
            self.build_model(self.model_dir)
            self.train()
            self.evaluate()
            self.save_model(self.model_name+".pth")
        except:
            traceback.print_exception()


    def set_transform(self):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform

    # 1️⃣ 데이터 로드 & 정규화
    def load_study_data(self):
        try:

            train_data = datasets.ImageFolder(self.train_dir, transform=self.transform)
            val_data = datasets.ImageFolder(self.val_dir, transform=self.transform)

            self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

            print(f"✅ Train classes: {train_data.classes}")
        except:
            traceback.print_exc()

    # 2️⃣ 모델 구성
    def build_model(self, load_path=None):

        try:
            #  """
            # load_path: 기존 학습된 모델(.pth) 경로
            # """
            # torch.hub.set_dir("models")
            self.model = models.resnet18(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False  # feature extractor 고정

            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
            self.model = self.model.to(self.device)

            load = f'{load_path}{self.model_name}.pth'
            print(f'load_path :: {load}')
            if load_path and os.path.exists(load):
                self.model.load_state_dict(torch.load(load, map_location=self.device))
                print(f"✅ 저장된 모델 불러오기 완료: {load}")
            else:
                print("✅ 모델 구성 완료 (새로 학습할 준비)")


        except:
            traceback.print_exc()





    # 3️⃣ 학습
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {avg_loss:.4f}")

    # 4️⃣ 검증
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"✅ Validation Accuracy: {100 * correct / total:.2f}%")

    # 5️⃣ 모델 저장
    def save_model(self, path="classifier.pth"):
        savePath = f'{self.model_dir}/{self.model_name}.pth'
        torch.save(self.model.state_dict(), savePath)
        print(f"✅ 모델 저장 완료: {path}")


    # 6️⃣ 단일 이미지 예측
    def predict_image(self, image_path):
        self.model.eval()

        img = Image.open(image_path).convert("RGB")
        print(f'img : {img}')
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x)
            label_idx = torch.argmax(pred, 1).item()

        return label_idx
