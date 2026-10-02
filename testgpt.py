import torch
from gpt2 import make_chunks, simple_attention


def test_make_chunks_shapes_and_shift():
    ids = list(range(1000))
    inputs, targets = make_chunks(ids, max_length=256, stride=256)
    assert inputs.shape == (3, 256)                       # starts at 0, 256, 512
    assert inputs[1, 0] == 256                            # chunks neither overlap nor skip tokens
    assert torch.equal(targets[:, :-1], inputs[:, 1:])    # targets = inputs shifted by one


def test_attention_keeps_shape_and_rows_sum_to_one():
    h = torch.randn(4, 256, 768)
    context, weights = simple_attention(h)
    assert weights.shape == (4, 256, 256)                 # one square score matrix per chunk
    assert context.shape == h.shape                       # same shape in and out
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 256))


def test_chunks_do_not_mix():
    h = torch.randn(2, 6, 3)
    context, _ = simple_attention(h)
    h2 = h.clone()
    h2[1] += 100                                          # change chunk 1 only
    context2, _ = simple_attention(h2)
    assert torch.allclose(context[0], context2[0])        # chunk 0 didn't notice


def test_matches_book_toy_example():
    toy = torch.tensor(
        [[0.43, 0.15, 0.89],   # Your
         [0.55, 0.87, 0.66],   # journey
         [0.57, 0.85, 0.64],   # starts
         [0.22, 0.58, 0.33],   # with
         [0.77, 0.25, 0.10],   # one
         [0.05, 0.80, 0.55]]   # step
    ).unsqueeze(0)                                        # [1, 6, 3]: a batch of one chunk
    context, _ = simple_attention(toy)
    assert torch.allclose(context[0, 1], torch.tensor([0.4419, 0.6515, 0.5683]), atol=1e-4)
    