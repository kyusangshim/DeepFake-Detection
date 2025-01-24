from copy import deepcopy
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train(train_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    count = 0.0
    with tqdm(total=len(train_loader), desc='Training', unit='batch') as progress_bar:
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            try:
                total_accuracy += roc_auc_score(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            except:
                pass
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1), 'accuracy': total_accuracy / (progress_bar.n + 1)})
    return total_loss / len(train_loader), total_accuracy / len(train_loader)

def validate(val_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc='Validation', unit='batch') as progress_bar:
            for inputs, labels in val_loader:
                outputs = model(inputs, inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                try:
                    total_accuracy += roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
                except:
                    pass
                progress_bar.update(1)
                progress_bar.set_postfix({'val_loss': total_loss / (progress_bar.n + 1), 'val_accuracy': total_accuracy / (progress_bar.n + 1)})
    return total_loss / len(val_loader), total_accuracy / len(val_loader)


def test(test_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing', unit='batch') as progress_bar:
            for inputs, labels in test_loader:
                outputs = model(inputs, inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                try:
                    total_accuracy += roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
                except:
                    pass
                progress_bar.update(1)
                progress_bar.set_postfix({'test_loss': total_loss / (progress_bar.n + 1), 'test_accuracy': total_accuracy / (progress_bar.n + 1)})
    return total_loss / len(test_loader), total_accuracy / len(test_loader)
    return total_loss / len(test_val_loader), total_accuracy / len(test_val_loader)
