import matplotlib.pyplot as plt
from PIL import Image
from src.recognizer import Recognizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == '__main__':
    recognizer = Recognizer(dataset_path='data/processed')

    test_img_path = 'data/raw/teste_mascarado.jpg'
    name, score = recognizer.recognize(test_img_path)

    if name:
        print(f"Reconhecido como: {name} (score: {score:.2f})")
    else:
        print("Nenhum rosto reconhecido.")

    test_img = Image.open(test_img_path)

    if name:
        db_img_path = os.path.join("data/processed", f"{name}.jpg")
        db_img = Image.open(db_img_path)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.title("Imagem de Teste")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(db_img)
        plt.title(f"Reconhecido como: {name}\nScore: {score:.2f}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    else:
        plt.imshow(test_img)
        plt.title("Nenhum rosto reconhecido")
        plt.axis("off")
        plt.show()
