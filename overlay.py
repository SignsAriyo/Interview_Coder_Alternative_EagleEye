import sys
import ctypes
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

# Win32 Constants
WDA_EXCLUDEFROMCAPTURE = 0x00000011
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x80000
LWA_COLORKEY = 0x00000001

class InvisibleOverlay(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 500, 250)

        # 1. Base flags (Topmost and frameless)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 2. Text styling
        self.label = QLabel("AI Copilot: Waiting for stream...", self)
        # Give text a slight background panel so it's readable on bright screens
        self.label.setStyleSheet("""
            color: #00FF00; 
            font-family: 'Consolas'; 
            font-size: 15px; 
            font-weight: bold;
            background-color: #121212;
            border-radius: 5px;
            padding: 8px;
        """)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.label)
        
        # 3. SET WINDOW TO SOLID MAGENETA
        # We do NOT use WA_TranslucentBackground anymore.
        self.setStyleSheet("background-color: #FF00FF;") 
        
        self._old_pos = None
        self.show()

        # 4. Apply Windows native transparent color key mask
        self.apply_chroma_key_cloak()

    def apply_chroma_key_cloak(self):
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            
            # Make the window structurally "Layered" in the Windows OS architecture
            current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style | WS_EX_LAYERED)

            # Convert RGB Magenta (255, 0, 255) to a Windows COLORREF structure (0x00FF00FF)
            # This tells Windows to key out ONLY the magenta background, leaving text untouched
            magenta_colorref = 0x00FF00FF
            user32.SetLayeredWindowAttributes(hwnd, magenta_colorref, 0, LWA_COLORKEY)

            # Block the rest of the window boundaries from screenshots entirely
            user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            print("Success! Windows 11 Chroma Cloak Active. No black box.")
        except Exception as e:
            print(f"Error applying Chroma Key: {e}")

    # --- Mouse Dragging Events ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self._old_pos is not None:
            delta = QPoint(event.globalPosition().toPoint() - self._old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self._old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self._old_pos = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = InvisibleOverlay()
    sys.exit(app.exec())