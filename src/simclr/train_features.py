import time
import torch

def train_features(model, epochs, trainloader, valloader, optimizer, scheduler, device, criterion):
    for epoch in range(epochs):
        start = time.time()
        model.train()
        train_loss = 0
        for xi, xj, _ in trainloader:
            
            xi = xi.to(device)
            xj = xj.to(device)
            
            optimizer.zero_grad()
            hi, hj, zi, zj = model(xi, xj)
            loss = criterion(zi, zj)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
        
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for xi, xj, targets in valloader:
                xi = xi.to(device)
                xj = xj.to(device)
            
                hi, hj, zi, zj = model(xi, xj)
                loss = criterion(zi, zj)
                test_loss += loss.item()
        
        train_loss = train_loss/len(trainloader)
        test_loss = test_loss/len(valloader)
        end = time.time()
        
        s = int(end - start)
        print(f"Epoch: [{epoch}], Learning Rate: {lr:.4f}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Time: {s//60:02}:{s%60:02}")