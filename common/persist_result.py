
def persist_training(epoch, acc, loss, tokens, lr):
    print("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f | Tokens / Sec: %7.1f | Learning Rate: %6.1e" % (epoch, acc, loss, tokens, lr))

def persist_validation(epoch, loss):
    print(f"epoch %f : loss = %f", epoch, loss)
