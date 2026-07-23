import argparse
import torch
from model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out/checkpoint.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--length", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)

    config = ckpt["config"]
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    model = GPT(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def encode(s):
        return [stoi[c] for c in s if c in stoi]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.length)[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    main()
