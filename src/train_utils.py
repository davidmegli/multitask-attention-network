import os
import time
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob

def create_run_folder(model_name):
    """ Crea una cartella unica per il run dentro 'models/' """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}_{timestamp}"
    run_path = os.path.join("models", run_name)
    os.makedirs(run_path, exist_ok=True)
    return run_path

def save_checkpoint(model, optimizer, epoch, val_loss, run_path, is_best=False):
    """ Salva un checkpoint con nome formattato """
    filename = "best_model.pth" if is_best else f"checkpoint_{epoch}.pth"
    checkpoint_path = os.path.join(run_path, filename)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint salvato: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_dir):
    """
    Carica l'ultimo checkpoint disponibile da una cartella.

    Args:
        model (torch.nn.Module): Il modello in cui caricare i pesi.
        optimizer (torch.optim.Optimizer): L'ottimizzatore in cui caricare lo stato.
        checkpoint_dir (str): Percorso della cartella dei checkpoint.

    Returns:
        int, float: Epoca da cui riprendere, miglior loss salvata.
    """
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Cartella '{checkpoint_dir}' non trovata.")
        return 0, float('inf')

    # Troviamo tutti i file checkpoint_*.pth
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    if not checkpoint_files:
        print(f"❌ Nessun checkpoint trovato in '{checkpoint_dir}'.")
        return 0, float('inf')

    # Ordiniamo i checkpoint per epoca e prendiamo l'ultimo
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  
    latest_checkpoint = checkpoint_files[-1]

    # Carichiamo il checkpoint
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_loss = checkpoint["val_loss"]

    print(f"✅ Checkpoint caricato: '{latest_checkpoint}' (Riprendo da Epoch {epoch+1}, Best Loss: {best_loss:.4f})")
    return epoch, best_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None,
                use_tensorboard=False, use_wandb=False, project_name="Training_Project",
                checkpoint_path=None, resume=False, early_stopping_patience=None):
    """
    Funzione di training con TensorBoard, Weights & Biases, checkpoint ordinati e early stopping.

    Args:
        model (torch.nn.Module): Modello da addestrare.
        train_loader (DataLoader): Dati di training.
        val_loader (DataLoader): Dati di validazione.
        criterion (torch.nn.Module): Funzione di perdita.
        optimizer (torch.optim.Optimizer): Ottimizzatore.
        num_epochs (int): Numero di epoche.
        device (torch.device, optional): CPU o GPU.
        use_tensorboard (bool): Abilita TensorBoard.
        use_wandb (bool): Abilita Weights & Biases.
        project_name (str): Nome del progetto su W&B.
        checkpoint_path (str, optional): Cartella per i checkpoint.
        resume (bool): Se True, riprende il training da checkpoint.
        early_stopping_patience (int, optional): Numero massimo di epoche senza miglioramento prima di fermare.

    Returns:
        torch.nn.Module: Modello con i migliori pesi salvati.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Creiamo una cartella per i checkpoint
    if checkpoint_path is None:
        checkpoint_path = create_run_folder(model.__class__.__name__)

    start_epoch = 0
    best_loss = float('inf')
    epochs_no_improve = 0  # Contatore per early stopping

    # Resume training
    if resume:
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint")]
        if checkpoint_files:
            latest_checkpoint = sorted(checkpoint_files)[-1]  # Prende l'ultimo checkpoint
            start_epoch, best_loss = load_checkpoint(model, optimizer, os.path.join(checkpoint_path, latest_checkpoint))

    # Inizializza TensorBoard e W&B
    writer = SummaryWriter() if use_tensorboard else None
    if use_wandb:
        wandb.init(project=project_name, config={"epochs": num_epochs, "optimizer": optimizer.__class__.__name__})
        wandb.watch(model, log="all")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Accuracy calculation (for classification tasks)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total * 100
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Logging
        if use_tensorboard:
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch + 1)

        if use_wandb:
            wandb.log({"Loss/Train": train_loss, "Loss/Validation": val_loss, "Accuracy/Validation": val_acc})

        # Salva il checkpoint per ogni epoca
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path, is_best=True)
        else:
            epochs_no_improve += 1

        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            print(f"\n🔴 Early stopping attivato! Nessun miglioramento per {early_stopping_patience} epoche consecutive.")
            break

    if use_tensorboard:
        writer.close()
    if use_wandb:
        wandb.finish()

    # Carichiamo il miglior modello
    best_model_path = os.path.join(checkpoint_path, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path)["model_state_dict"])
    return model
