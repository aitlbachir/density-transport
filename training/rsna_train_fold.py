"""Train one fold of a breast-cancer image classifier on RSNA 512x512 PNGs and write out-of-fold scores.
Usage: python train_fold.py --meta folds.csv --img_dir DIR --fold 0 --out OUT [--epochs 10] [--smoke]
"""
import argparse, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import timm
from PIL import Image


class DS(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, train):
        self.p = [os.path.join(img_dir, f"{a}_{b}.png") for a, b in zip(df.patient_id, df.image_id)]
        self.y = df.cancer.to_numpy().astype(np.float32); self.train = train

    def __len__(self): return len(self.p)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(Image.open(self.p[i]).convert("L"), dtype=np.float32) / 255.0)[None]
        if self.train:
            if random.random() < .5: x = x.flip(-1)
            if random.random() < .5: x = x.flip(-2)
            if random.random() < .5:                                     # small affine
                ang, sc = random.uniform(-10, 10), random.uniform(0.9, 1.1)
                th = torch.tensor([[sc * np.cos(np.radians(ang)), -sc * np.sin(np.radians(ang)), random.uniform(-.05, .05)],
                                   [sc * np.sin(np.radians(ang)), sc * np.cos(np.radians(ang)), random.uniform(-.05, .05)]], dtype=torch.float32)
                g = F.affine_grid(th[None], x[None].shape, align_corners=False)
                x = F.grid_sample(x[None], g, align_corners=False)[0]
            x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)).clamp(0, 1)
        x = (x - 0.5) / 0.25
        return x.repeat(3, 1, 1), self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True); ap.add_argument("--img_dir", required=True)
    ap.add_argument("--fold", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--neg_ratio", type=float, default=10)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="efficientnet_b0"); ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed + a.fold); np.random.seed(a.seed + a.fold); torch.manual_seed(a.seed + a.fold)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; amp = dev == "cuda"
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.meta)
    tr, te = df[df.fold != a.fold].reset_index(drop=True), df[df.fold == a.fold].reset_index(drop=True)
    assert not set(tr.patient_id) & set(te.patient_id), "patient leakage across folds"
    if a.smoke:
        tr = pd.concat([tr[tr.cancer == 1].head(32), tr[tr.cancer == 0].head(96)]); te = te.head(64)
    pos, neg = tr[tr.cancer == 1], tr[tr.cancer == 0]
    model = timm.create_model(a.backbone, pretrained=not a.no_pretrained, num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(pos) * (1 + a.neg_ratio) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    t0 = time.time()
    for ep in range(a.epochs):
        n_neg = min(len(neg), int(round(a.neg_ratio * len(pos))))
        ep_df = pd.concat([pos, neg.sample(n_neg, random_state=a.seed * 100 + ep)]).sample(frac=1, random_state=ep)
        L = torch.utils.data.DataLoader(DS(ep_df, a.img_dir, True), batch_size=a.bs, shuffle=False,
                                        num_workers=a.workers, drop_last=True, pin_memory=amp)
        model.train(); tot, nb = 0.0, 0
        for x, y in L:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if nb < steps_per_epoch: sched.step()
            tot += loss.item(); nb += 1
        print(f"fold {a.fold} epoch {ep} loss {tot / max(nb, 1):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    torch.save(model.state_dict(), f"{a.out}/model.pt")
    # out-of-fold inference, average of original and horizontally flipped image
    model.eval(); L = torch.utils.data.DataLoader(DS(te, a.img_dir, False), batch_size=2 * a.bs, num_workers=a.workers)
    outs = []
    with torch.no_grad():
        for x, _ in L:
            x = x.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                s = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(x.flip(-1)).float())) / 2
            outs.append(s.squeeze(1).cpu().numpy())
    te["score"] = np.concatenate(outs)
    te[["patient_id", "image_id", "laterality", "fold", "score"]].to_csv(f"{a.out}/oof.csv", index=False)
    print(f"fold {a.fold} done: {len(te)} images scored in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
