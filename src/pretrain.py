import os
from datetime import datetime

import torch

from src.data import PreTrainData
from src.model import ModelConfig, Model
from src.tokenizer import Tokenizer


@torch.no_grad()
def loss_estimate(model, data, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iter)
        for k in range(config.eval_iter):
            X, Y = data.get_batch(use_type=split, seq_size=200, batch_size=2)
            X = X.to(config.device)
            Y = Y.to(config.device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_model(model, config, save_dir="checkpoints", model_name=None):
    """
    save model state and configs
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"gpt2_base_{timestamp}"

    model_path = os.path.join(save_dir, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.save(model.state_dict(), os.path.join(model_path, "model_state_dict.pth"))
    torch.save(model, os.path.join(model_path, "full_model.pth"))


    config_dict = {
        'n_embd': config.n_embd,
        'n_head': config.n_head,
        'n_ctx': config.n_ctx,
        'n_layer': config.n_layer,
        'device': config.device,
        'max_iter': config.max_iter,
        'lr': config.lr,
        'interval': config.interval,
        'eval_iter': config.eval_iter,
        'vocab_size': model.tokenizer.vocab_size,
        'max_token': model.max_token
    }
    # save hyperparameters
    torch.save(config_dict, os.path.join(model_path, "config.pth"))

    # save model architecture
    model_info = {
        'model_type': 'GPT2',
        'architecture': 'transformer',
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'save_time': datetime.now().isoformat()
    }
    torch.save(model_info, os.path.join(model_path, "model_info.pth"))

    print(f"model saved to: {model_path}")
    print(f"model info: {model_info}")

    return model_path


def load_model(model_path, config):
    """
    load model from checkpoint
    """
    config_dict = torch.load(os.path.join(model_path, "config.pth"), map_location=config.device)

    print(f"\n previous training configs:")
    print(f"model: {config_dict.get('n_layer', 'N/A')} layers, "
          f"{config_dict.get('n_embd', 'N/A')}D of embeddings, "
          f"{config_dict.get('n_head', 'N/A')} heads")
    print(f"   training configs: lr={config_dict.get('lr', 'N/A')}, max_iter={config_dict.get('max_iter', 'N/A')}")
    print(f"   device: {config_dict.get('device', 'N/A')}")

    warnings = []
    if config_dict.get('n_embd') != config.n_embd:
        warnings.append(f"n_embd: {config_dict.get('n_embd')} vs {config.n_embd}")
    if config_dict.get('n_head') != config.n_head:
        warnings.append(f"n_head: {config_dict.get('n_head')} vs {config.n_head}")
    if config_dict.get('n_layer') != config.n_layer:
        warnings.append(f"n_layer: {config_dict.get('n_layer')} vs {config.n_layer}")

    if warnings:
        print("NotCompatibleException: ")
        for warning in warnings:
            print(f"   {warning}")

    model = Model(
        n_ctx=config_dict['n_ctx'],
        max_token=config_dict['max_token'],
        n_embd=config_dict['n_embd'],
        n_hc=config_dict['n_head'],
        p=config.p,
        n_layer=config_dict['n_layer'],
        device=config.device
    )

    state_dict = torch.load(os.path.join(model_path, "model_state_dict.pth"), map_location=config.device)
    model.load_state_dict(state_dict)

    model_info = torch.load(os.path.join(model_path, "model_info.pth"), map_location=config.device)

    print(f"model loaded from: {model_path}")
    print(f"all parameter count: {model_info['total_params']:,}")
    print(f"save time: {model_info['save_time']}")

    return model, config_dict, model_info


if __name__ == '__main__':
    """
    base generative pre-train model training script
    """
    torch.manual_seed(10384)
    tokenizers = Tokenizer()
    config = ModelConfig()

    if config.continue_pretrain:
        model, _, _ = load_model(config.model_checkpoints_path, config)
    else:
        model = Model(
            config.n_ctx, 200,
            config.n_embd, config.n_head, config.p,
            n_layer=config.n_layer, device=config.device
        )
    data = PreTrainData(0.9, config.dataset_path)
    prompt = torch.tensor(tokenizers.encode("hello world!")).unsqueeze(0).to(config.device)

    print("========================= training =========================")

    model.train()
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    adamW = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adamW, mode='min', factor=0.5, patience=3)

    for i in range(config.max_iter):
        if i % config.interval == 0:
            out = loss_estimate(model, data, config)
            print(f"iter: {i}, out:{out}")
            print(f"model generate: {tokenizers.decode(model.gen(prompt)[0].tolist())}")

            current_val_loss = out['val']
            scheduler.step(current_val_loss)

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                # patience_counter = 0
                print(f"best val loss: {current_val_loss:.4f}")
            # else:
            #     patience_counter += 1
            #     print(f"val loss can't be descend! ==> ({patience_counter}/{patience})")
            #
            #     if patience_counter >= patience:
            #         print("========================= stop training =========================")
            #         break

        X, Y = data.get_batch(seq_size=1024, batch_size=16)
        X = X.to(device=config.device)
        Y = Y.to(device=config.device)
        logits, loss = model(X, Y)
        adamW.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adamW.step()

    print("========================= finish =========================")
    print(f"final best val loss: {best_val_loss:.4f}")

    final_out = loss_estimate(model, data, config)
    print(f"final val loss: {final_out}")

    model_path = save_model(model, config)

