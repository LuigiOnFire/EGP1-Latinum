import model
import torch
import test

print(f"Using {device} as the device")

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f? [{current:>5d}/{size:>5d}]")   

    datetime = today.strftime("%d-%m-%Y_%H-%M")
    torch.save(model.state_dict(), f"model_{datetime}.pth")
    print("Saved PyTorch Model State to model.pth")

if __name__ = "main":
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    )

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-*30")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
