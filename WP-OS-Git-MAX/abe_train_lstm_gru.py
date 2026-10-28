#!/usr/bin/env python3
"""Train real PyTorch LSTM/GRU multi-label heads on earlier Power Lottery draws.

Research only. No inference on an untrained network; no 4-model fusion claim.
Uses the previous 20 draws per training sample; never trains on a future target.
"""
import argparse
import json
import pathlib
from wp_os_git_max import load

def run(draws, epochs=5, seed=42, hidden=24):
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise RuntimeError("PyTorch required: pip install torch") from e
    if len(draws) < 25:
        raise ValueError("At least 25 completed draws required for 20-draw training")
    if epochs < 1 or epochs > 200 or hidden < 1:
        raise ValueError("Invalid training hyperparameters")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    def encode(draw):
        vector = [0.0] * 46
        for n in draw["first"]:
            vector[n - 1] = 1.0
        vector[38 + draw["second"] - 1] = 1.0
        return vector
    rows = [encode(d) for d in draws]
    x = torch.tensor([rows[i-20:i] for i in range(20, len(rows))], dtype=torch.float32)
    yf = torch.tensor([rows[i][:38] for i in range(20, len(rows))], dtype=torch.float32)
    ys = torch.tensor([draws[i]["second"] - 1 for i in range(20, len(rows))], dtype=torch.long)
    inp = torch.tensor([rows[-20:]], dtype=torch.float32)
    output = {}
    for name, layer in (("lstm", nn.LSTM), ("gru", nn.GRU)):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = layer(46, hidden, batch_first=True)
                self.head_first = nn.Linear(hidden, 38)
                self.head_second = nn.Linear(hidden, 8)
            def forward(self, seq):
                z, _ = self.rnn(seq)
                return self.head_first(z[:, -1]), self.head_second(z[:, -1])
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=.005)
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            f, s = net(x)
            loss = nn.functional.binary_cross_entropy_with_logits(f, yf) + nn.functional.cross_entropy(s, ys)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            f, s = net(inp)
            first = torch.sigmoid(f[0]).tolist()
            second = torch.softmax(s[0], dim=-1).tolist()
        output[name] = {
            "first": {str(n): first[n-1] for n in range(1,39)},
            "second": {str(n): second[n-1] for n in range(1,9)}
        }
    return {
        "through_draw": draws[-1]["draw_id"], "models": output,
        "mode": "trained_local_lstm_gru_only",
        "training_samples": len(draws) - 20,
        "features_per_sample": 20,
        "epochs": epochs, "seed": seed,
        "chronos_2": "not_executed", "t0_gamma": "not_executed",
        "five_model_fusion_executed": False,
        "note": "In-sample training only; not validated predictive advantage."
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    result = run(load(args.csv), epochs=args.epochs, seed=args.seed)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("LSTM/GRU real training output saved; 4-model ensemble unavailable.")

if __name__ == "__main__":
    main()
