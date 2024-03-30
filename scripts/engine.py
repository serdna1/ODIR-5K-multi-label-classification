import copy
import torch
from tqdm.auto import tqdm
from metrics import compute_challenge_metrics

def train_step(model, 
               dataloader, 
               loss_fn, 
               optimizer,
               device):
    model.train()

    train_loss = 0
    all_y = torch.tensor([], dtype=torch.float32).to(device)
    all_probs = torch.tensor([], dtype=torch.float32).to(device)

    for ((X_left, X_right), y) in dataloader:
        X_left, X_right, y = X_left.to(device), X_right.to(device), y.to(device)

        logits = model(X_left, X_right)
        probs = torch.sigmoid(logits)

        loss = loss_fn(logits, y)
        train_loss += loss.item() 

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        all_y = torch.cat((all_y, y))
        all_probs = torch.cat((all_probs, probs))

    all_y = all_y.detach().cpu().numpy()
    all_probs = all_probs.detach().cpu().numpy()
    
    train_loss = train_loss / len(dataloader)
    kappa, f1, auc, final_score = compute_challenge_metrics(all_y, all_probs)
    
    return train_loss, kappa, f1, auc, final_score

def val_step(model, 
             dataloader, 
             loss_fn,
             device):
    model.eval() 

    val_loss = 0
    all_y = torch.tensor([], dtype=torch.float32).to(device)
    all_probs = torch.tensor([], dtype=torch.float32).to(device)

    with torch.inference_mode():
        for ((X_left, X_right), y) in dataloader:
            X_left, X_right, y = X_left.to(device), X_right.to(device), y.to(device)

            logits = model(X_left, X_right)
            probs = torch.sigmoid(logits)

            loss = loss_fn(logits, y)
            val_loss += loss.item()

            all_y = torch.cat((all_y, y))
            all_probs = torch.cat((all_probs, probs))

    all_y = all_y.detach().cpu().numpy()
    all_probs = all_probs.detach().cpu().numpy()
    
    val_loss = val_loss / len(dataloader)
    kappa, f1, auc, final_score = compute_challenge_metrics(all_y, all_probs)

    return val_loss, kappa, f1, auc, final_score, all_y, all_probs

def train(model, 
          train_dataloader, 
          val_dataloader, 
          train_loss_fn,
          val_loss_fn,
          optimizer,
          scheduler,
          epochs,
          stopper,
          device,
          writer):
    results = {
        'train_loss': [],
        'train_kappa': [],
        'train_f1': [],
        'train_auc': [],
        'train_final_score': [],
        'val_loss': [],
        'val_kappa': [],
        'val_f1': [],
        'val_auc': [],
        'val_final_score': [],
    }
    
    model.to(device)

    for epoch in tqdm(range(epochs)):
        (train_loss,
         train_kappa,
         train_f1,
         train_auc,
         train_final_score) = train_step(model=model,
                                         dataloader=train_dataloader,
                                         loss_fn=train_loss_fn,
                                         optimizer=optimizer,
                                         device=device)
        (val_loss,
         val_kappa,
         val_f1,
         val_auc,
         val_final_score,
         _,
         _) = val_step(model=model,
                       dataloader=val_dataloader,
                       loss_fn=val_loss_fn,
                       device=device)

        print(
            f'Ep: {epoch} | '
            f't_loss: {train_loss:.4f} | '
            f't_kappa: {train_kappa:.4f} | '
            f't_f1: {train_f1:.4f} | '
            f't_auc: {train_auc:.4f} | '
            f't_final: {train_final_score:.4f} | '
            f'v_loss: {val_loss:.4f} | '
            f'v_kappa: {val_kappa:.4f} | '
            f'v_f1: {val_f1:.4f} | '
            f'v_auc: {val_auc:.4f} | '
            f'v_final: {val_final_score:.4f}'
        )

        results['train_loss'].append(train_loss)
        results['train_kappa'].append(train_kappa)
        results['train_f1'].append(train_f1)
        results['train_auc'].append(train_auc)
        results['train_final_score'].append(train_final_score)
        results['val_loss'].append(val_loss)
        results['val_kappa'].append(val_kappa)
        results['val_f1'].append(val_f1)
        results['val_auc'].append(val_auc)
        results['val_final_score'].append(val_final_score)

        # Track experiments with a writer
        if writer:
            writer.add_scalars(main_tag='Loss', 
                               tag_scalar_dict={'train_loss': train_loss,
                                                'val_loss': val_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag='Kappa', 
                               tag_scalar_dict={'train_kappa': train_kappa,
                                                'val_kappa': val_kappa}, 
                               global_step=epoch)
            writer.add_scalars(main_tag='F1', 
                               tag_scalar_dict={'train_f1': train_f1,
                                                'val_f1': val_f1}, 
                               global_step=epoch)
            writer.add_scalars(main_tag='AUC', 
                               tag_scalar_dict={'train_auc': train_auc,
                                                'val_auc': val_auc}, 
                               global_step=epoch)
            writer.add_scalars(main_tag='FINAL', 
                               tag_scalar_dict={'train_final_score': train_final_score,
                                                'val_final_score': val_final_score}, 
                               global_step=epoch)

            writer.close()

        # stopper needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        stopper(val_loss, model)
        
        if stopper.early_stop:
            print("Early stopping")
            break

        if scheduler is not None:
            # Update lr for next epoch
            scheduler.step()

    best_model = copy.deepcopy(model)
    
    # load the last checkpoint with the best model
    best_model.load_state_dict(torch.load(stopper.path))

    return model, best_model, results