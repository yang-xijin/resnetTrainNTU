import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


# train and val
def train_val(epoch, model, train_loader, len_train, val_loader, len_val, criterion, optimizer, device):
    torch.cuda.empty_cache()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0
    model.to(device)
    fit_time = time.time()
    for i in range(epoch):
        print("-------第{}轮训练开始-------".format(i + 1))
        since = time.time()
        running_loss = 0
        training_acc = 0
        # 训练
        with tqdm(total=len(train_loader)) as pbar:
            for img, label in train_loader:
                model.train()
                optimizer.zero_grad()
                img = img.to(device)
                label = label.to(device)
                # forward
                output = model(img)
                loss = criterion(output, label)
                predict_t = torch.max(output, dim=1)[1]

                # 优化器调优 backward
                loss.backward()
                optimizer.step()  # update weights

                running_loss += loss.item()
                training_acc += torch.eq(predict_t, label).sum().item()
                pbar.update(1)

        # 测试步骤
        model.eval()
        val_losses = 0
        validation_acc = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for img, label in val_loader:
                    img = img.to(device)
                    label = label.to(device)
                    output = model(img)

                    # loss
                    loss = criterion(output, label)
                    predict_v = torch.max(output, dim=1)[1]

                    val_losses += loss.item()
                    validation_acc += torch.eq(predict_v, label).sum().item()
                    pb.update(1)

            # calculate mean for each batch
            train_loss.append(running_loss / len_train)
            val_loss.append(val_losses / len_val)

            train_acc.append(training_acc / len_train)
            val_acc.append(validation_acc / len_val)

            torch.save(model, "./result/last.pth")
            if best_acc < (validation_acc / len_val):
                torch.save(model, "./result/best.pth")

            print("Epoch:{}/{}..".format(i + 1, epoch),
                  "Train Acc: {:.3f}..".format(training_acc / len_train),
                  "Val Acc: {:.3f}..".format(validation_acc / len_val),
                  "Train Loss: {:.3f}..".format(running_loss / len_train),
                  "Val Loss: {:.3f}..".format(val_losses / len_val),
                  "Time: {:.2f}s".format((time.time() - since)))

    history = {'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def plot_loss(x, history):
    plt.plot(x, history['val_loss'], label='val', marker='o')
    plt.plot(x, history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('./weight/loss.png')
    plt.show()


def plot_acc(x, history):
    plt.plot(x, history['train_acc'], label='train_acc', marker='x')
    plt.plot(x, history['val_acc'], label='val_acc', marker='x')
    plt.title('Acc per epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('./weight/acc.png')
    plt.show()