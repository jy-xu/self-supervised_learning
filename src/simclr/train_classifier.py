import time
import torch

def train_classifier(model, epochs, trainloader, valloader, optimizer, scheduler, device, criterion, feature_model=None):
    for epoch in range(epochs):
        start = time.time()
        if feature_model is not None:
            feature_model.eval()
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, _, targets in trainloader:
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if feature_model is not None:
                with torch.no_grad():
                    rep, _ = feature_model(inputs)
                    rep = rep.detach()
            
            optimizer.zero_grad()
            if feature_model is not None:
                outputs = model(rep)
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss/len(trainloader)
        train_acc = correct / total
        
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, _, targets in valloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                if feature_model is not None:
                    rep, _ = feature_model(inputs)
                    rep = rep.detach()
                    outputs = model(rep)
                else:
                    outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        
        test_loss = test_loss/len(valloader)
        test_acc = correct / total
        end = time.time()
        
        s = int(end - start)
        print(f"Epoch: [{epoch}], Learning Rate: {lr:.4f}, Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.2%}, "
              f"Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2%}, "
              f"Time: {s//60:02}:{s%60:02}")