import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
from typing import Tuple, Dict
import json
from sklearn.metrics import  classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


class SingleImageDataset(Dataset):  #Tek bir MRI görüntüsünü işlemek için 
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform  
        self.image = Image.open(image_path).convert('RGB')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.image)
        return image


class AlzheimerDataModule:  #preprocessing bölümü
    
    def __init__(self, data_dir: str, batch_size: int = 32):
        self.data_dir = data_dir #data klasoru
        self.batch_size = batch_size 
        self.transform = self._create_transforms() 
        self.train_loader, self.val_loader, self.test_loader = self._setup_data_loaders()
        self.class_names = ['Hafif', 'Orta', 'Normal', 'Çok Hafif']  # Türkçe sınıf isimleri

    def _create_transforms(self):
        
        return transforms.Compose([ 
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

    def _setup_data_loaders(self):  #datasetler işlenmesi için belleğe yüklenir.
        
        train_dataset = ImageFolder(f'{self.data_dir}/train', transform=self.transform)
        val_dataset = ImageFolder(f'{self.data_dir}/test', transform=self.transform)
        test_dataset = ImageFolder(f'{self.data_dir}/test', transform=self.transform)

        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True),
            DataLoader(val_dataset, batch_size=self.batch_size),
            DataLoader(test_dataset, batch_size=self.batch_size)
        )

    def prepare_single_image(self, image_path: str): 
        dataset = SingleImageDataset(image_path, self.transform)
        return DataLoader(dataset, batch_size=1)


class AlzheimerModel(nn.Module): #Model
    
    def __init__(self, num_classes= 4):
        super().__init__()
        self.weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.model = torchvision.models.resnet18(weights=self.weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):  #forward propagation
        return self.model(x) 

   

class AlzheimerTrainer:

    def __init__(self, model: nn.Module, data_module: AlzheimerDataModule,
                 learning_rate: float = 0.001, num_epochs: int = 10): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data_module = data_module
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} #her epoch için değerler bu listlere kaydedilir.

    def predict_single_image(self, image_path: str) -> str:
        self.load_model('alzheimer_model.pth')
        self.model.eval()

        loader = self.data_module.prepare_single_image(image_path)

        with torch.no_grad(): #gradyanlalr bellek tasarrufu icin iptal
            for inputs in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1) #en yuksek olasilik
                prediction = predicted.item()

                # Tahmin sonucunu Türkçe olarak döndür
                return self.data_module.class_names[prediction]

    def epoch(self) -> Tuple[float, float]:

        self.model.train()
        running_loss = 0.0 #toplam kayıp değeri
        correct = 0 #doğru tahminlerin sayısı
        total = 0   #İşlenen toplam örnek sayısı (batch boyutu toplamı)

        for inputs, labels in tqdm(self.data_module.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs) #Forward pass
            loss = self.criterion(outputs, labels)
            loss.backward() #Backward propagation
            self.optimizer.step() #ağırlıklar güncellenir.

            #İstatistikler güncellenir.
            running_loss += loss.item() 
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() #tahmin edilen sınıf gerçek sınıfa eşit mi?

        return running_loss/len(self.data_module.train_loader), 100.*correct/total

    def validate(self, loader: DataLoader) -> Tuple[float, float]: 
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                #validation verisini kullan
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss/len(loader), 100.*correct/total

    def train(self):
        best_val_acc = 0

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.epoch() #model önce train modunda(ağırlıklar güncellenir)
            val_loss, val_acc = self.validate(self.data_module.val_loader) #model evaluate(değerlendirme) modunda

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if val_acc > best_val_acc: 
                best_val_acc = val_acc
                self.save_model('alzheimer_model.pth')

        with open('training_history.json', 'w') as f:
            json.dump(self.history, f)

    def save_model(self, path: str):
        torch.save(self.model.model.state_dict(), path)

    def load_model(self, path: str):
        try: 
            state_dict = torch.load(path)
            self.model.model.load_state_dict(state_dict)
        except Exception as e: #hata cikarsa
            print(f"Model yükleme hatası: {e}")
            print("Alternatif yükleme yöntemi deneniyor...")
            try: 
                # Alternatif yükleme yöntemi
                state_dict = torch.load(path)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                self.model.model.load_state_dict(new_state_dict)
                print("Model başarıyla yüklendi!")
                
            except Exception as e: #yine cikarsa yok artik yapacak bi sey
                print(f"Alternatif yükleme de başarısız: {e}")

    def test(self):
        self.load_model('alzheimer_model.pth')
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
             for inputs, labels in self.data_module.test_loader:
                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                 outputs = self.model(inputs)
                 _, predicted = outputs.max(1)

                 y_true.extend(labels.cpu().numpy())
                 y_pred.extend(predicted.cpu().numpy())

        test_loss, test_acc = self.validate(self.data_module.test_loader)
        print(f'\nTest Accuracy: {test_acc:.2f}%')
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.data_module.class_names))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.data_module.class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

        return test_acc


def main():

    try:
        clear_screen()
        time.sleep(1)
        print("\nSistem başlatılıyor...")

        for _ in tqdm(range(20), desc="Modüller Yükleniyor"):
            time.sleep(0.1)

        time.sleep(1)
        clear_screen()
        print("\n")
        print("╔════════════════════════════════════════╗")
        print("║          Alzheimer MRI Analiz          ║")
        print("╚════════════════════════════════════════╝")
        print("\n")
        print("Model yükleniyor...")
        data_module = AlzheimerDataModule('alzheimer_dataset')
        model = AlzheimerModel()
        trainer = AlzheimerTrainer(model, data_module)
        
        if os.path.exists('alzheimer_model.pth'):  # Eğer önceden eğitilmiş model varsa yükle
            print("\nÖnceden eğitilmiş model kontrol ediliyor...")
            for _ in tqdm(range(5), desc="Model Yükleniyor"): 
                time.sleep(0.2)
            trainer.load_model('alzheimer_model.pth')
            print("Model başarıyla yüklendi!")

        time.sleep(1)
        clear_screen()

        while True:
            print("\n╔════════════════════════════════╗")
            print("║           ANA MENÜ             ║")
            print("╚════════════════════════════════╝")
            print("\nLütfen yapmak istediğiniz işlemi seçin:")
            print("1. Yeni MRI görüntüsü analiz et")
            print("2. Modeli yeniden eğit")
            print("3. Test sonuçlarını görüntüle")
            print("4. Çıkış")

            choice = input("\nSeçiminiz (1-4): ")
            if choice == '1':
                clear_screen()
                print("\n╔════════════════════════════════╗")
                print("║         MRI ANALİZİ            ║")
                print("╚════════════════════════════════╝")
                image_path = input("\nLütfen MRI görüntüsünün dosya yolunu girin: ")
                if os.path.exists(image_path):
                    print("\nGörüntü analiz ediliyor...")
                    for _ in tqdm(range(10), desc="İşleniyor"): #1-10 aralığında yükleme çizgisi oluşturulur.
                        time.sleep(0.2)
                    result = trainer.predict_single_image(image_path)
                    print(f"\nSonuç: {result}")
                    input("\nAna menüye dönmek için ENTER'a basın...")
                    clear_screen()
                else:
                    print("\nHATA: Belirtilen dosya bulunamadı!")
                    time.sleep(2)
                    clear_screen()

            elif choice == '2':
                clear_screen()
                print("\n╔════════════════════════════════╗")
                print("║         MODEL EĞİTİMİ          ║")
                print("╚════════════════════════════════╝")
                print("\nModel eğitimi başlatılıyor...")
                print("CUDA yüklü." if torch.cuda.is_available() else "CUDA yüklü değil.")
                print(f"Cihaz: {trainer.device}")
                trainer.train() 
                print("Eğitim tamamlandı!")
                input("\nAna menüye dönmek için ENTER'a basın...")
                clear_screen()

            elif choice == '3':
                clear_screen()
                print("\n╔════════════════════════════════╗")
                print("║       TEST SONUÇLARI           ║")
                print("╚════════════════════════════════╝")
                print("\nTest sonuçları hesaplanıyor...")
                test_acc = trainer.test()
                input("\nAna menüye dönmek için ENTER'a basın...")
                clear_screen()

            elif choice == '4':
                clear_screen()
                print("\nProgram kapatılıyor...")
                for _ in tqdm(range(5), desc="Kapatılıyor"):
                    time.sleep(0.2)
                break

            else:
                print("\nGeçersiz seçim! Lütfen tekrar deneyin.")
                time.sleep(2)
                clear_screen()

    except Exception as e:
        print(f"\nBir hata oluştu: {e}")
        print("Program kapatılıyor...")
        time.sleep(3)

if __name__ == "__main__":
    main()