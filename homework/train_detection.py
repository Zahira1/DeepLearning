import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
from .models import Detector, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric

def train(num_epochs=30, lr=1e-3, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = tb.SummaryWriter("logs")

    # 1. load data
    train_data = load_data("drive_data/train", shuffle=True,  batch_size=batch_size)
    val_data   = load_data("drive_data/val",   shuffle=False, batch_size=batch_size)

    # 2. model, loss, optimizer
    model     = Detector().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    depth_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(num_epochs):
        # 3. train
        model.train()
        for batch in train_data:
            img= batch["image"].to(device)
            track, depth = batch["track"].to(device), batch["depth"].to(device)
            logits, raw_depth = model(img)       # forward pass
            loss = loss_fn(logits, track)  # compute loss
            if raw_depth is not None:
                depth_loss = depth_loss_fn(raw_depth, depth)
                loss += depth_loss
            optimizer.zero_grad()     # clear old gradients
            loss.backward()           # backprop
            optimizer.step()          # update weights
            global_step += 1
            logger.add_scalar("train/loss", loss.item(), global_step)  # log loss

        # 4. validate
        model.eval()
        metric = DetectionMetric()
        with torch.inference_mode():
            for batch in val_data:
                img, track, depth = batch["image"].to(device), batch["track"].to(device), batch["depth"].to(device)
                pred, pred_depth = model.predict(img)
                metric.add(pred, track, pred_depth, depth)



        m = metric.compute()
        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"acc={m['accuracy']:.3f}  iou={m['iou']:.3f}  "
              f"depth_err={m['abs_depth_error']:.3f}  tp_depth_err={m['tp_depth_error']:.3f}")
        logger.add_scalar("val/iou", m["iou"], global_step)
        logger.add_scalar("val/depth_err", m["abs_depth_error"], global_step)

        logger.add_scalar("val/accuracy", m["accuracy"], global_step)

    # 5. save
    save_model(model)
    logger.close()

if __name__ == "__main__":
    train()
