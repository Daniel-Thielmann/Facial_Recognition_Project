from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def extract_features(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            img_embedding = model(img_cropped.unsqueeze(0).to(device))
            return img_embedding.detach().cpu().numpy()
        else:
            print(f"Nenhum rosto detectado em {img_path}")
            return None
    except Exception as e:
        print(f"Erro ao processar {img_path}: {e}")
        return None
