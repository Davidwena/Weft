def dc_cost(kin, kout, parties, comparisons, depths, alarm=False):
    N = kin + 2 * kout
    if not kin or not kout:
        return dict(
            rounds=2,
            preprocessing_bits=0,
            output_gate_preprocessing_bits=194,
            open_bits=2,
            shuffle_bits=0,
            online_bits=2,
        )
    h = (N - 1).bit_length()
    tail = sum(35 + ell for ell in range(1, h))
    mask_bits = N * (283 + 4 * (147 + h) + 4 * tail)
    shuffle_bits = N * (283 + 2 * (147 + h) + 2 * tail)
    pre = (
        1983 * comparisons[0]
        + 3143 * sum(comparisons[1:])
        + 2 * N * h * 807
        + 2 * N * (h - 1) * 96
        + mask_bits
    )
    online = (
        82 * comparisons[0]
        + 90 * sum(comparisons[1:])
        + 2 * N * h * 135
        + 2 * N * sum(131 + ell for ell in range(1, h))
        + 2 * N * (147 + h)
        + 2
    )

    if alarm:
        from run_primitives import costs

        a, b, _ = costs("msb_fresh", 32)
        pre += a
        online += b
    rounds = (
        3 * depths[0] + 3 * max(depths[1:]) + 2 * parties + (h - 1) * (parties + 4) + 4 + 2 * alarm
    )
    return dict(
        rounds=rounds,
        preprocessing_bits=pre,
        open_bits=online,
        shuffle_bits=shuffle_bits,
        online_bits=online + shuffle_bits,
        positional_mask_bits=mask_bits,
        output_gate_preprocessing_bits=194,
    )
