from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import traceback
import torch
import gc

print("GUI başlatılıyor...")
print(f"PyQt5 versiyonu: {PYQT_VERSION_STR}")
print(f"Python versiyonu: {sys.version}")
print(f"Torch versiyonu: {torch.__version__}")
print(f"CUDA kullanılabilir: {torch.cuda.is_available()}")

try:
    print("Ana model modülü import ediliyor...")
    from main import AlzheimerDataModule, AlzheimerModel, AlzheimerTrainer
    print("Model modülü başarıyla import edildi")
except Exception as e:
    print(f"Model import hatası: {str(e)}")
    print(f"Detaylı hata: {traceback.format_exc()}")
    sys.exit(1)

class DropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # Ana pencereye referans
        self.setAlignment(Qt.AlignCenter)
        self.setText("\n\nMRI Görüntüsünü Buraya Sürükleyin\nveya\nTıklayarak Seçin")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 10px;
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 20px;
                font-size: 16px;
            }
        """)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 300)

    def get_main_window(self):
        # Ana pencereyi bul
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, AlzheimerGUI):
                return parent
            parent = parent.parent()
        return None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        main_window = self.get_main_window()
        if main_window and event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            main_window.process_image(file_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            main_window = self.get_main_window()
            if main_window:
                file_name, _ = QFileDialog.getOpenFileName(
                    self,
                    "MRI Görüntüsü Seç",
                    "",
                    "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
                )
                if file_name:
                    main_window.process_image(file_name)

class AlzheimerGUI(QMainWindow):
    def create_left_panel(self):
        left_panel = QGroupBox("Görüntü Yükleme")
        left_layout = QVBoxLayout(left_panel)

        # DropLabel oluştururken self'i parent olarak geçir
        self.drop_label = DropLabel(self)
        left_layout.addWidget(self.drop_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #666;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        left_layout.addWidget(self.progress)

        return left_panel

class AlzheimerGUI(QMainWindow):
    def __init__(self):
        print("AlzheimerGUI başlatılıyor...")
        super().__init__()
        self.model = None
        self.trainer = None
        self.data_module = None
        self.init_ui()
        self.load_model_thread = ModelLoadThread(self)
        self.load_model_thread.finished.connect(self.on_model_loaded)
        self.load_model_thread.error.connect(self.on_model_error)
        self.load_model_thread.start()

    def init_ui(self):
        print("UI başlatılıyor...")
        try:
            self.setWindowTitle("MRI Alzheimer Teşhis Sistemi")
            self.setMinimumSize(900, 700)

            # Ana widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)

            # Başlık
            title_widget = self.create_title_widget()
            main_layout.addWidget(title_widget)

            # Ana içerik
            content_widget = QWidget()
            content_layout = QHBoxLayout(content_widget)

            # Sol panel
            left_panel = self.create_left_panel()
            content_layout.addWidget(left_panel)

            # Sağ panel
            right_panel = self.create_right_panel()
            content_layout.addWidget(right_panel)

            main_layout.addWidget(content_widget)

            # Durum çubuğu
            self.statusBar().showMessage('Başlatılıyor...')
            self.statusBar().setStyleSheet("""
                QStatusBar {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    padding: 5px;
                }
            """)

            # Tema ayarları
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #34495e;
                }
                QWidget {
                    color: #ecf0f1;
                }
                QGroupBox {
                    border: 2px solid #455a64;
                    border-radius: 5px;
                    margin-top: 1ex;
                    font-size: 14px;
                    padding: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px 0 3px;
                }
            """)

            print("UI başarıyla oluşturuldu")
        except Exception as e:
            print(f"UI başlatma hatası: {str(e)}")
            print(f"Detaylı hata: {traceback.format_exc()}")
            raise

    def create_title_widget(self):
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)

        title = QLabel("MRI Alzheimer Teşhis Sistemi")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #ecf0f1;
                padding: 20px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel(" MRI Alzheimer Teşhis Sistemi")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #bdc3c7;
                padding-bottom: 20px;
            }
        """)
        subtitle.setAlignment(Qt.AlignCenter)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        return title_widget

    def create_left_panel(self):
        left_panel = QGroupBox("Görüntü Yükleme")
        left_layout = QVBoxLayout(left_panel)

        self.drop_label = DropLabel()
        left_layout.addWidget(self.drop_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #666;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        left_layout.addWidget(self.progress)

        return left_panel

    def create_right_panel(self):
        right_panel = QGroupBox("Analiz Sonuçları")
        right_layout = QVBoxLayout(right_panel)

        self.result_label = QLabel("Henüz analiz yapılmadı")
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                padding: 15px;
                color: #ecf0f1;
                background-color: #34495e;
                border-radius: 5px;
            }
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.result_label)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: none;
                padding: 10px;
            }
        """)
        right_layout.addWidget(self.details_text)

        return right_panel

    def on_model_loaded(self):
        print("Model başarıyla yüklendi")
        self.statusBar().showMessage('Model hazır')

    def on_model_error(self, error_msg):
        print(f"Model yükleme hatası: {error_msg}")
        QMessageBox.critical(self, "Model Yükleme Hatası", error_msg)

    def process_image(self, image_path):
        if self.trainer is None:
            QMessageBox.warning(self, "Uyarı", "Model henüz yüklenmedi, lütfen bekleyin.")
            return

        print(f"Görüntü işleme başlatılıyor: {image_path}")
        try:
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.statusBar().showMessage('Analiz yapılıyor...')

            self.process_thread = ImageProcessThread(self.trainer, image_path)
            self.process_thread.finished.connect(self.on_process_finished)
            self.process_thread.error.connect(self.on_process_error)
            self.process_thread.start()

            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.drop_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.drop_label.setPixmap(scaled_pixmap)

        except Exception as e:
            self.on_process_error(str(e))

    def on_process_finished(self, result):
        try:
            self.result_label.setText(f"Teşhis: {result}")
            self.details_text.setText(
                f"Analiz Detayları:\n\n"
                f"• Teşhis: {result}\n"
                f"• Analiz Tarihi: {QDateTime.currentDateTime().toString('dd.MM.yyyy hh:mm')}\n"
            )
            self.progress.setVisible(False)
            self.statusBar().showMessage('Analiz tamamlandı')
        except Exception as e:
            print(f"Sonuç gösterme hatası: {str(e)}")

    def on_process_error(self, error_msg):
        self.progress.setVisible(False)
        self.statusBar().showMessage('Hata oluştu!')
        QMessageBox.critical(self, "Analiz Hatası", error_msg)

class ModelLoadThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, gui):
        super().__init__()
        self.gui = gui

    def run(self):
        try:
            print("Model yükleme thread'i başlatıldı")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            device = torch.device("cpu")
            torch.set_num_threads(1)

            self.gui.data_module = AlzheimerDataModule('alzheimer_dataset')
            self.gui.model = AlzheimerModel()
            self.gui.trainer = AlzheimerTrainer(self.gui.model, self.gui.data_module)

            if os.path.exists('best_alzheimer_model.pth'):
                state_dict = torch.load('best_alzheimer_model.pth', map_location=device)
                self.gui.trainer.model.load_state_dict(state_dict)

            self.finished.emit()
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            self.error.emit(str(e))

class ImageProcessThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, trainer, image_path):
        super().__init__()
        self.trainer = trainer
        self.image_path = image_path

    def run(self):
        try:
            print(f"Görüntü işleme thread'i başlatıldı: {self.image_path}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            result = self.trainer.predict_single_image(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            print(f"Görüntü işleme hatası: {str(e)}")
            self.error.emit(str(e))

def main():
    print("Ana program başlatılıyor...")
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        app = QApplication(sys.argv)

        sys._excepthook = sys.excepthook
        def exception_hook(exctype, value, traceback):
            print(f"Beklenmeyen hata: {exctype}, {value}")
            sys._excepthook(exctype, value, traceback)
            sys.exit(1)
        sys.excepthook = exception_hook

        print("GUI penceresi oluşturuluyor...")
        window = AlzheimerGUI()
        window.show()

        print("Uygulama döngüsü başlatılıyor...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Ana program hatası: {str(e)}")
        print(f"Detaylı hata: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    print("Program başlatılıyor...")
    main()