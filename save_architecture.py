import torch
from src.segnet_mtan import SegNetMTAN
from src.segnet import SegNet
from src.original_segnet_mtan import SegNet as OriginalSegNetMTAN

# Se disponibile, usa la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lista di tuple (nome_modello, classe_modello)
models = [
    ("SegNetMTAN", SegNetMTAN),
    ("SegNet", SegNet),
    ("OriginalSegNetMTAN", OriginalSegNetMTAN)
]

for name, ModelClass in models:
    # Istanzia e sposta su GPU
    model = ModelClass().to(device)
    
    # Cattura la rappresentazione testuale del modello
    model_str = str(model)
    
    # Scrive su file
    with open(f"{name}.txt", "w") as f:
        f.write(model_str)
    with open(f"{name}_summary.txt", "w") as f:
        f.write(model.summary())

    print(f"Architettura salvata in {name}.txt")
