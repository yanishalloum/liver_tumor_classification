import pytest

torch = pytest.importorskip("torch")

from livertumor.training.model import (  # noqa: E402
    NUM_CLASSES,
    build_loss,
    build_unet,
    count_parameters,
)


def test_unet_output_shape():
    model = build_unet(in_channels=5, base_channels=8, depth=4)
    x = torch.randn(2, 5, 64, 64)  # batch of 2 patches, 5 slices each
    assert model(x).shape == (2, NUM_CLASSES, 64, 64)  # one score map per class


def test_default_model_size():
    # 32-64-128-256-512 channels: a few million weights, fine for a 16 GB GPU
    assert 1e6 < count_parameters(build_unet()) < 2e7


def test_loss_rewards_correct_predictions():
    loss_fn = build_loss()
    label = torch.zeros(1, 1, 16, 16, dtype=torch.long)
    label[..., 4:12, 4:12] = 1  # liver square
    label[..., 6:8, 6:8] = 2  # small tumor inside
    perfect = torch.nn.functional.one_hot(label[:, 0], NUM_CLASSES).permute(0, 3, 1, 2) * 20.0
    wrong = perfect.roll(shifts=1, dims=1)  # every pixel scored as the wrong class
    assert loss_fn(perfect, label) < 0.05
    assert loss_fn(wrong, label) > 1.0


def test_can_overfit_one_batch():
    """Sanity check: a network that cannot memorise 2 patches has a bug somewhere."""
    torch.manual_seed(0)
    model = build_unet(in_channels=1, base_channels=8, depth=3)
    loss_fn = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    image = torch.randn(2, 1, 32, 32)
    label = torch.zeros(2, 1, 32, 32, dtype=torch.long)
    label[:, :, 8:24, 8:24] = 1
    label[:, :, 12:16, 12:16] = 2
    image[label > 0] += 2.0  # make the structures visible in the input

    first = None
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(image), label)
        loss.backward()
        optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first
