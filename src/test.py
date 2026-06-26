import torch


def test():
    print(torch.cuda.is_bf16_supported())


if __name__ == "__main__":
    test()
