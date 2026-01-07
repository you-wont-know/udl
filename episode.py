from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from cnn import CNN

# Static list of weight decay hyperparameters to try.
WEIGHT_DECAYS = [1e-4, 1e-3, 1e-2, 2.5e-2, 5e-2]


def evaluate_loss(model, data_loader, device, criterion, use_dropout=False):
    if use_dropout:
        model.train()
    else:
        model.eval()
    loss_total = 0.0
    total = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_size = yb.numel()
            loss_total += loss.item() * batch_size
            total += batch_size
    return loss_total / total if total else float("inf")

def evaluate_accuracy(model, data_loader, device, use_dropout, dropout_samples):
    if use_dropout:
        model.train()
    else:
        model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model.forward_probs(xb, dropout_samples).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total if total else 0.0

def evaluate_accuracy_and_rmse_mean(model, data_loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    sum_sq = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mean, _ = model(xb)
            preds = mean.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
            one_hot = torch.nn.functional.one_hot(yb, num_classes=num_classes).to(mean.dtype)
            diff = mean - one_hot
            sum_sq += diff.pow(2).sum().item()
            count += diff.numel()
    rmse = (sum_sq / count) ** 0.5 if count else 0.0
    acc = correct / total if total else 0.0
    return acc, rmse


def evaluate_rmse_mean(model, data_loader, device, num_classes):
    _, rmse = evaluate_accuracy_and_rmse_mean(model, data_loader, device, num_classes)
    return rmse

def train_to_convergence(
    model : torch.nn.Module,
    train_loader,
    val_loader,
    device,
    weight_decay,
    max_epochs=200,
    patience=30,
    val_use_dropout=False,
):
    model.train()
    criterion = torch.nn.CrossEntropyLoss() # classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_params = None
    epochs_since_improve = 0
    for epoch in range(max_epochs):
        # Do an epoch
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Evaluate on the validation set
        avg_val_loss = evaluate_loss(
            model,
            val_loader,
            device,
            criterion,
            use_dropout=val_use_dropout,
        )

        # Stop if we are getting worse
        if best_val_loss > avg_val_loss: # improvement
            best_val_loss = avg_val_loss
            best_params = deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break
    if best_params is not None:
        model.load_state_dict(best_params)
    return best_params


def train_and_test(
    model_clas,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    dropout_samples,
    batch_size,
    device,
    use_dropout=True
):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    best_val_acc = -1.0
    best_state = None
    best_weight_decay = None # for logging/debugging

    for weight_decay in WEIGHT_DECAYS:

        # Train a model with this weight decay
        model = model_clas().to(device)
        train_to_convergence(
            model,
            train_loader,
            val_loader,
            device,
            weight_decay,
            val_use_dropout=use_dropout,
        )

        # Pick the best model according to validation accuracy
        val_acc = evaluate_accuracy(model, val_loader, device, use_dropout, dropout_samples)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())
            best_weight_decay = weight_decay

    # Recreate the best model
    best_model = model_clas().to(device)
    if best_state is not None:
        best_model.load_state_dict(best_state)

    test_acc = evaluate_accuracy(best_model, test_loader, device, use_dropout, dropout_samples)
    return best_model, test_acc


def train_variational_to_convergence(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs=10000, # 10k just in case
    patience=40,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    best_val_elbo = float("-inf")
    best_params = None
    epochs_since_improve = 0
    for e in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            elbo = model.elbo(xb, yb)
            loss = -elbo
            loss.backward()
            optimizer.step()

        model.eval()
        val_elbo = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                val_elbo += model.elbo(xb, yb).item()
        if val_elbo > best_val_elbo:
            best_val_elbo = val_elbo
            best_params = deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break


    if best_params is not None:
        model.load_state_dict(best_params)
    return best_val_elbo


class FeatureExtractor(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.net = nn.Sequential(*list(cnn.net.children())[:-1])

    def forward(self, x):
        feats = self.net(x.unsqueeze(1))
        norm = feats.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return feats / norm

def build_feature_extractor(cnn_model):
    extractor = FeatureExtractor(cnn_model)
    extractor.eval()
    return extractor

def select_best_candidate(candidates, score_fn):
    best_score = float("inf")
    best_model = None
    for candidate in candidates:
        score = score_fn(candidate)
        if score < best_score:
            best_score = score
            best_model = deepcopy(candidate)
    return best_model


def evaluate_pool_scores(
    pool,
    model,
    acquisition_function,
    batch_size,
    device,
    T,
    use_dropout=True, # useful, but acquisition functions might override this
):
    pool_x, pool_y = pool
    if pool_x.numel() == 0:
        return torch.empty(0)

    pool_loader = DataLoader(TensorDataset(pool_x, pool_y), batch_size=batch_size, shuffle=False)

    if hasattr(model, "forward_probs"): # if CNN
        if use_dropout:
            model.train()
        else:
            model.eval()
    scores_list = []
    with torch.no_grad():
        for xb, _ in pool_loader:
            xb = xb.to(device)
            scores = acquisition_function(xb, model, T)
            scores_list.append(scores.view(-1).detach().cpu())

    return torch.cat(scores_list, dim=0)


def evaluate_pool_scores_two_stage(
    pool,
    model,
    acquisition_function,
    batch_size,
    device,
    T,
    top_k=500,
    use_dropout=True, # useful, but acquisition functions might override this
):
    pool_x, pool_y = pool
    if pool_x.numel() == 0:
        return torch.empty(0)

    pool_loader = DataLoader(TensorDataset(pool_x, pool_y), batch_size=batch_size, shuffle=False)

    if hasattr(model, "forward_probs"): # if CNN
        if use_dropout:
            model.train()
        else:
            model.eval()
    scores_list = []
    with torch.no_grad():
        for xb, _ in pool_loader:
            xb = xb.to(device)
            scores = acquisition_function(xb, model, T=1)
            scores_list.append(scores.view(-1).detach().cpu())

    coarse_scores = torch.cat(scores_list, dim=0)
    k = min(top_k, coarse_scores.shape[0])
    top_indices = torch.topk(coarse_scores, k).indices

    top_x = pool_x.index_select(0, top_indices.to(pool_x.device))
    top_y = pool_y.index_select(0, top_indices.to(pool_y.device))
    top_loader = DataLoader(TensorDataset(top_x, top_y), batch_size=batch_size, shuffle=False)

    refined_scores_list = []
    with torch.no_grad():
        for xb, _ in top_loader:
            xb = xb.to(device)
            scores = acquisition_function(xb, model, T=T)
            refined_scores_list.append(scores.view(-1).detach().cpu())

    refined_scores = torch.cat(refined_scores_list, dim=0)
    full_scores = torch.full((pool_x.shape[0],), float("-inf"))
    full_scores[top_indices] = refined_scores
    return full_scores


def redistribute_pool(train, pool, scores, k):
    # Take top k from pool according to scores
    # and change train and pool datasets accordingly
    pool_x, pool_y = pool

    selected_indices = torch.topk(scores, k).indices
    selected_indices = selected_indices.to(pool_x.device)

    selected_x = pool_x.index_select(0, selected_indices)
    selected_y = pool_y.index_select(0, selected_indices)

    print("Selected ys: ", selected_y)
    print("top scores:",  torch.topk(scores, k).values)

    # Add in the selected to train
    train_x, train_y = train
    new_train_x = torch.cat([train_x, selected_x], dim=0)
    new_train_y = torch.cat([train_y, selected_y], dim=0)

    # Remove the selected from pool
    keep_mask = torch.ones(pool_x.shape[0], dtype=torch.bool, device=pool_x.device)
    keep_mask[selected_indices] = False
    new_pool_x = pool_x[keep_mask]
    new_pool_y = pool_y[keep_mask]

    del train_x
    del train_y
    del pool_x
    del pool_y

    # Free up space by removing the old tensors

    return (new_train_x, new_train_y), (new_pool_x, new_pool_y)


def episode(
    model_clas,
    train,
    val,
    pool,
    test,
    acquisition_function,
    evaluate_pool_scores_fn,
    batch_size,
    device,
    dropout_samples, # for evaluating test/val predictions
    acquisition_T,
    use_dropout=True, # during testing
    model_kind="cnn",
    model_params=None,
    s_candidates=None,
    k=10,
):
    if model_params is None:
        model_params = {}

    if model_kind == "cnn":
        # Train on the current dataset
        model, test_acc = train_and_test(
            model_clas,
            train[0], train[1], 
            val[0], val[1], 
            test[0], test[1],
            dropout_samples,
            batch_size,
            device,
            use_dropout=use_dropout,
        )
        test_rmse = None
    else:
        # Pre-train a CNN to obtain the feature extractor.
        base_model, _ = train_and_test(
            CNN,
            train[0], train[1],
            val[0], val[1],
            test[0], test[1],
            dropout_samples,
            batch_size,
            device,
            use_dropout=use_dropout,
        )
        base_model.eval()
        feature_extractor = build_feature_extractor(base_model).to(device)

        if model_kind in {"analytical", "variational"}:
            val_loader = DataLoader(TensorDataset(val[0], val[1]), batch_size=batch_size, shuffle=False)
            train_loader = None
            if model_kind == "variational":
                train_loader = DataLoader(TensorDataset(train[0], train[1]), batch_size=batch_size, shuffle=True)

            def train_candidate(s):
                candidate = model_clas(s=s, phi=feature_extractor, **model_params).to(device)
                if model_kind == "analytical":
                    candidate.fit(train[0], train[1])
                else:
                    train_variational_to_convergence(candidate, train_loader, val_loader, device)
                return candidate

            candidates = [train_candidate(s) for s in s_candidates]
            model = select_best_candidate(
                candidates,
                lambda m: evaluate_rmse_mean(m, val_loader, device, num_classes=10),
            )

        model.eval()
        test_loader = DataLoader(TensorDataset(test[0], test[1]), batch_size=batch_size, shuffle=False)
        test_acc, test_rmse = evaluate_accuracy_and_rmse_mean(model, test_loader, device, num_classes=10)

    # Select new samples from the pool
    scores = evaluate_pool_scores_fn(
        pool,
        model,
        acquisition_function,
        batch_size,
        device,
        acquisition_T,
        use_dropout=use_dropout,
    )
    new_train, new_pool = redistribute_pool(train, pool, scores, k)
    return new_train, val, new_pool, test, test_acc, test_rmse
