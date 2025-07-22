import os, random, numpy as np, torch, config
from datetime import datetime
from pathlib import Path
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from dataset.MultiModalDataset import MultiModalDataset
from logger import TBLogger
from trade import trading
from model import CnnTaFusion

# ───────── reproducibility ─────────
torch.manual_seed(42);
np.random.seed(42);
random.seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────── training loop ───────────
def _run_training(
    in_channels: int,
    include_sentiment: bool,
    log_suffix: str,
    comment: str,
    last_day_only: bool = False,
):
    model = config.model(
        in_channels=in_channels,
        window_length=config.sequence_len,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(config.class_weights, device=device))

    finance_train_dir = list(Path(config.train_dir).glob(config.sentiment_ticker + "*.csv"))[0]
    finance_test_dir = list(Path(config.test_dir).glob(config.sentiment_ticker + "*.csv"))[0]

    train_ds = MultiModalDataset(
        str(finance_train_dir),
        config.sentiment_dir,
        config.indicators,
        include_sentiment=include_sentiment,
        last_day_sentiment=last_day_only,
        sequence_len=config.sequence_len,
    )
    train_ld = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          pin_memory=True, drop_last=True)

    test_ds = MultiModalDataset(
        str(finance_test_dir),
        config.sentiment_dir,
        config.indicators,
        include_sentiment=include_sentiment,
        last_day_sentiment=last_day_only,
        sequence_len=config.sequence_len,
    )
    test_ld = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                         pin_memory=True)

    run_name = f"{config.run_name}_{log_suffix}"
    logger = TBLogger(config.record_dir, run_name, comment=comment)

    for epoch in trange(config.max_epochs, desc="Epochs"):
        # =================== TRAIN =================== #
        model.train()
        ep_loss, p_all, y_all = 0.0, [], []

        for _, _, imgs, lbls in tqdm(train_ld, desc="Train", leave=False):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            outs = model(imgs)
            loss = crit(outs, lbls)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            p_all.extend(torch.argmax(outs, 1).cpu().tolist())
            y_all.extend(torch.argmax(lbls, 1).cpu().tolist())

        logger.log_epoch("train", epoch, ep_loss / len(train_ld),
                         np.array(y_all), np.array(p_all))

        # =================== EVAL ==================== #
        model.eval()
        tot_loss, batches = 0.0, 0
        p_all, y_all = [], []
        by_symbol = {}

        with torch.no_grad():
            for ts, closes, imgs, lbls in tqdm(test_ld, desc="Eval", leave=False):
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)

                tot_loss += crit(outs, lbls).item()
                batches += 1

                preds = torch.argmax(outs, 1).cpu().tolist()
                labs = torch.argmax(lbls, 1).cpu().tolist()

                p_all.extend(preds)
                y_all.extend(labs)

                bucket = by_symbol.setdefault(config.sentiment_ticker, {"ts": [], "cl": [], "pr": []})
                bucket["ts"].extend(ts.cpu().tolist())
                bucket["cl"].extend(closes.cpu().tolist())
                bucket["pr"].extend(preds)

        logger.log_epoch("eval", epoch, tot_loss / batches,
                         np.array(y_all), np.array(p_all))

        trade_res = [trading(v["ts"], v["cl"], v["pr"])[0]
                     for v in by_symbol.values()]
        logger.log_trading(epoch, trade_res)

    logger.close()


def _run_finance_then_freeze(log_suffix: str, comment: str) -> None:
    """Train a 2-channel model, first on finance only then freeze finance
    convolution weights and continue training with sentiment."""

    model = config.model(in_channels=2, window_length=config.sequence_len).to(device)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(config.class_weights, device=device))

    finance_train_dir = list(Path(config.train_dir).glob(config.sentiment_ticker + "*.csv"))[0]
    finance_test_dir = list(Path(config.test_dir).glob(config.sentiment_ticker + "*.csv"))[0]

    train_ds = MultiModalDataset(
        str(finance_train_dir),
        config.sentiment_dir,
        config.indicators,
        include_sentiment=True,
        sequence_len=config.sequence_len,
    )
    train_ld = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          pin_memory=True, drop_last=True)

    test_ds = MultiModalDataset(
        str(finance_test_dir),
        config.sentiment_dir,
        config.indicators,
        include_sentiment=True,
        sequence_len=config.sequence_len,
    )
    test_ld = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                         pin_memory=True)

    run_name = f"{config.run_name}_{log_suffix}"
    logger = TBLogger(config.record_dir, run_name, comment=comment)

    # ── Stage A: train using only finance channel ─────────────────
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in trange(config.max_epochs, desc="Fin only"):
        model.train()
        ep_loss, p_all, y_all = 0.0, [], []
        for _, _, imgs, lbls in tqdm(train_ld, desc="Train", leave=False):
            imgs[:, 1] = 0  # zero-out sentiment channel
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            outs = model(imgs)
            loss = crit(outs, lbls)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            p_all.extend(torch.argmax(outs, 1).cpu().tolist())
            y_all.extend(torch.argmax(lbls, 1).cpu().tolist())

        logger.log_epoch("train", epoch, ep_loss / len(train_ld), np.array(y_all), np.array(p_all))

        # ---- eval ----
        model.eval()
        tot_loss, batches = 0.0, 0
        p_all, y_all = [], []
        by_symbol = {}
        with torch.no_grad():
            for ts, closes, imgs, lbls in tqdm(test_ld, desc="Eval", leave=False):
                imgs[:, 1] = 0
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                tot_loss += crit(outs, lbls).item(); batches += 1
                preds = torch.argmax(outs, 1).cpu().tolist(); labs = torch.argmax(lbls, 1).cpu().tolist()
                p_all.extend(preds); y_all.extend(labs)
                bucket = by_symbol.setdefault(config.sentiment_ticker, {"ts": [], "cl": [], "pr": []})
                bucket["ts"].extend(ts.cpu().tolist())
                bucket["cl"].extend(closes.cpu().tolist())
                bucket["pr"].extend(preds)

        logger.log_epoch("eval", epoch, tot_loss / batches, np.array(y_all), np.array(p_all))
        trade_res = [trading(v["ts"], v["cl"], v["pr"])[0] for v in by_symbol.values()]
        logger.log_trading(epoch, trade_res)

    # ── freeze finance channel weights ─────────────────────────────
    def zero_finance_grad(grad):
        grad[:, 0] = 0
        return grad

    model.conv1.weight.register_hook(zero_finance_grad)
    model.conv1.bias.requires_grad_(False)

    # new optimizer for unfrozen params only
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    # ── Stage B: train with sentiment channel ─────────────────────
    for epoch in trange(config.max_epochs, desc="Fin frozen"):
        model.train(); ep_loss, p_all, y_all = 0.0, [], []
        for _, _, imgs, lbls in tqdm(train_ld, desc="Train", leave=False):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            outs = model(imgs)
            loss = crit(outs, lbls)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            p_all.extend(torch.argmax(outs, 1).cpu().tolist())
            y_all.extend(torch.argmax(lbls, 1).cpu().tolist())

        offset = config.max_epochs + epoch
        logger.log_epoch("train", offset, ep_loss / len(train_ld), np.array(y_all), np.array(p_all))

        # ---- eval ----
        model.eval(); tot_loss, batches = 0.0, 0; p_all, y_all = [], []; by_symbol = {}
        with torch.no_grad():
            for ts, closes, imgs, lbls in tqdm(test_ld, desc="Eval", leave=False):
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                tot_loss += crit(outs, lbls).item(); batches += 1
                preds = torch.argmax(outs, 1).cpu().tolist(); labs = torch.argmax(lbls, 1).cpu().tolist()
                p_all.extend(preds); y_all.extend(labs)
                bucket = by_symbol.setdefault(config.sentiment_ticker, {"ts": [], "cl": [], "pr": []})
                bucket["ts"].extend(ts.cpu().tolist()); bucket["cl"].extend(closes.cpu().tolist()); bucket["pr"].extend(preds)

        logger.log_epoch("eval", offset, tot_loss / batches, np.array(y_all), np.array(p_all))
        trade_res = [trading(v["ts"], v["cl"], v["pr"])[0] for v in by_symbol.values()]
        logger.log_trading(offset, trade_res)

    logger.close()


def _run_fusion_vector(log_suffix: str, comment: str) -> None:
    """Train a model that fuses finance features with a sentiment MLP."""

    finance_train_dir = list(Path(config.train_dir).glob(config.sentiment_ticker + "*.csv"))[0]
    finance_test_dir = list(Path(config.test_dir).glob(config.sentiment_ticker + "*.csv"))[0]

    train_ds = MultiModalDataset(
        str(finance_train_dir),
        config.sentiment_dir,
        config.indicators,
        include_sentiment=True,
        num_topics=config.num_topics,
        sequence_len=config.sequence_len,
    )
    train_ld = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )

    test_ds = MultiModalDataset(
        str(finance_test_dir),
        config.sentiment_dir,
        config.indicators,
        include_sentiment=True,
        num_topics=config.num_topics,
        sequence_len=config.sequence_len,
    )
    test_ld = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    model = CnnTaFusion(
        num_topics=train_ds.num_topics,
        window_length=config.sequence_len,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    crit = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(config.class_weights, device=device)
    )

    run_name = f"{config.run_name}_{log_suffix}"
    logger = TBLogger(config.record_dir, run_name, comment=comment)

    for epoch in trange(config.max_epochs, desc="Fusion"):
        model.train(); ep_loss, p_all, y_all = 0.0, [], []
        for _, _, imgs, lbls in tqdm(train_ld, desc="Train", leave=False):
            fin = imgs[:, 0:1].to(device, non_blocking=True)
            sent = imgs[:, 1].to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            outs = model(fin, sent)
            loss = crit(outs, lbls)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            p_all.extend(torch.argmax(outs, 1).cpu().tolist())
            y_all.extend(torch.argmax(lbls, 1).cpu().tolist())

        logger.log_epoch("train", epoch, ep_loss / len(train_ld), np.array(y_all), np.array(p_all))

        # ---- eval ----
        model.eval(); tot_loss, batches = 0.0, 0; p_all, y_all = [], []; by_symbol = {}
        with torch.no_grad():
            for ts, closes, imgs, lbls in tqdm(test_ld, desc="Eval", leave=False):
                fin = imgs[:, 0:1].to(device)
                sent = imgs[:, 1].to(device)
                lbls = lbls.to(device)
                outs = model(fin, sent)
                tot_loss += crit(outs, lbls).item(); batches += 1
                preds = torch.argmax(outs, 1).cpu().tolist(); labs = torch.argmax(lbls, 1).cpu().tolist()
                p_all.extend(preds); y_all.extend(labs)
                bucket = by_symbol.setdefault(config.sentiment_ticker, {"ts": [], "cl": [], "pr": []})
                bucket["ts"].extend(ts.cpu().tolist()); bucket["cl"].extend(closes.cpu().tolist()); bucket["pr"].extend(preds)

        logger.log_epoch("eval", epoch, tot_loss / batches, np.array(y_all), np.array(p_all))
        trade_res = [trading(v["ts"], v["cl"], v["pr"])[0] for v in by_symbol.values()]
        logger.log_trading(epoch, trade_res)

    logger.close()


def train():
    # ---- Stage 1: finance-only training ----
    _run_training(in_channels=1, include_sentiment=False,
                  log_suffix="finance_only", comment="finance-only")

    # ---- Stage 2: full multimodal training ----
    _run_training(in_channels=2, include_sentiment=True,
                  log_suffix="multimodal", comment="multi-modal")

    # ---- Stage 3: finance pretrain then freeze ----
    _run_finance_then_freeze(log_suffix="fin_freeze", comment="fin-freeze")

    # ---- Stage 4: fusion with sentiment MLP ----
    _run_fusion_vector(log_suffix="fusion_vec", comment="fusion-vector")

    # ---- Stage 5: sentiment last-day only ----
    _run_training(
        in_channels=2,
        include_sentiment=True,
        log_suffix="last_day_sent",
        comment="last-day-sentiment",
        last_day_only=True,
    )


if __name__ == "__main__":
    tickers = getattr(config, "sentiment_ticker_list", [config.sentiment_ticker])
    base_name = getattr(config, "run_name_base", config.run_name)
    for t in tickers:
        config.sentiment_ticker = t
        config.sentiment_dir = os.path.join(
            config.working_dir,
            "data_sentim",
            "preprocessed",
            f"{t}.csv",
        )
        config.run_name = f"{base_name}_{t}-{datetime.now().strftime('%m_%d_%H_%M')}"
        train()
    print("✓ Training complete.")
