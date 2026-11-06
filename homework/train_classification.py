import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
from .models import Classifier, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric

def train(num_epochs=30, lr=1e-3, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = tb.SummaryWriter("logs")

    # 1. load data
    train_data = load_data("classification_data/train", shuffle=True,  batch_size=batch_size)
    val_data   = load_data("classification_data/val",   shuffle=False, batch_size=batch_size)

    # 2. model, loss, optimizer
    model     = Classifier().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(num_epochs):
        # 3. train
        model.train()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            logits = model(img)       # forward pass
            loss = loss_fn(logits, label)  # compute loss
            optimizer.zero_grad()     # clear old gradients
            loss.backward()           # backprop
            optimizer.step()          # update weights
            global_step += 1
            logger.add_scalar("train/loss", loss.item(), global_step)  # log loss

        # 4. validate
        model.eval()
        metric = AccuracyMetric()
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                pred = model.predict(img)
                metric.add(pred, label)

        val_acc = metric.compute()["accuracy"]
        print(f"Epoch {epoch+1}/{num_epochs}  val_acc={val_acc:.3f}")
        logger.add_scalar("val/accuracy", val_acc, global_step)

    # 5. save
    save_model(model)
    logger.close()

if __name__ == "__main__":
    train()
