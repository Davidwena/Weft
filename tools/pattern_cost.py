from run_primitives import costs


def validate_pattern(c, row):
    pattern = c["pattern"]
    Kx = c["kout"] if pattern == "P4" else c["kin"]
    Ky = c["kout"] if pattern == "P2" else c["kin"]
    N = Kx + 2 * Ky
    P = Kx * Ky
    alert = c.get("output") == "alarm"
    alarm_k = 64 if pattern == "P2" else 32
    extra_pre, extra_online = costs("msb_fresh", alarm_k)[:2] if alert else (0, 0)
    if c.get("operator", "nl") == "nl":
        if pattern == "P2":
            pre = 2588 * P + 576 * Ky + 194
            online = 58 * P + 46 * Ky + 2
            rounds = 5
        elif pattern == "P3":
            pass
        else:
            pre = 5148 * P + 194
            online = 69 * P + 2
            rounds = 2
        if pattern == "P3":

            def interval(k):
                from run_primitives import shape

                full, bits = shape(k - 1)
                d = min(range(1, 10), key=lambda d: shape(k - 1, d)[0])
                B = (k - 2 + d) // d
                table = B * (1 << d)
                return 2 * full - table, 2 * bits

            ta, tb = interval(c.get("kt", 32))
            aa, ab = interval(64)
            pre = (ta + 451) * P + (aa + 898) * Ky + 194
            online = (tb + 3) * P + (ab + 48) * Ky + 2
            rounds = 5
        assert row["operator_preprocessing_bits"] == pre + extra_pre, (
            pattern,
            "NL pre",
            row["operator_preprocessing_bits"],
            pre + extra_pre,
        )
        assert row["nl"]["contribution_bits"] == online + extra_online, (
            pattern,
            "NL online",
            row["nl"]["contribution_bits"],
            online + extra_online,
        )
        assert row["nl"]["open_rounds"] == rounds + 2 * alert
        return
    d = row["dc"]
    h = (N - 1).bit_length() if N else 0
    if not N or (pattern != "P3" and (not Kx or not Ky)):
        return
    sortstats = [(64, d["sorts"][0])]
    if pattern == "P3":
        sortstats.append((max(2, h + 1), d["restore_sorts"][0]))
        base_pre = 3519 * Ky
        online = 83 * Ky + 2 + 384 * N
        mask = 2 * (224 + 160) * N
        shuffle = (224 + 160) * N
        rounds = 3 * sum(s["depth"] for _, s in sortstats) + 2 * c["n"] + 7
    else:
        money = 72
        sortstats.extend([(money, d["sorts"][1]), (money, d["sorts"][2])])
        degree = pattern == "P2"
        base_pre = 2 * N * h * (872 if degree else 807) + 2 * N * (h - 1) * 96
        online = 2 * N * h * (136 if degree else 135) + 2
        initial = 355 if degree else 451
        amount = (235 if degree else 171) + h
        tail = sum((195 if degree else 35) + ell for ell in range(1, h))
        mask = 2 * N * (initial + 2 * amount + 2 * tail)
        shuffle = N * (initial + 2 * amount + 2 * tail)
        online += N * (initial + 2 * amount) + 2 * N * sum(
            96 + (195 if degree else 35) + ell for ell in range(1, h)
        )

        if degree:
            sortstats.extend([(max(2, h + 1), s) for s in d["restore_sorts"]])
            mask += 2 * N * 2 * 160
            shuffle += N * 2 * 160
            online += N * 2 * 160 + 46 * Ky
            base_pre += 576 * Ky
            extra_depth = max(s["depth"] for s in d["restore_sorts"])
        else:
            sortstats.append((96, d["counterparty_sort"]))
            mask += 2 * N * 547
            shuffle += N * 547
            online += N * 547 + 74 * (N - 1) + N
            base_pre += 2367 * (N - 1) + 97 * N
            extra_depth = d["counterparty_sort"]["depth"]
        rounds = (
            3 * (d["sorts"][0]["depth"] + max(s["depth"] for s in d["sorts"][1:]) + extra_depth)
            + (h + 2) * c["n"]
            + 4 * h
            + 4
        )
    comparison_pre = 0
    for k, s in sortstats:
        a, b, _ = costs("msb_fresh", k)
        comparison_pre += a * s["comparisons"]
        online += (b + 1) * s["comparisons"]
    expected = base_pre + comparison_pre + mask + extra_pre
    assert d["preprocessing_consumed_bits"] == expected, (
        pattern,
        "DC pre",
        d["preprocessing_consumed_bits"],
        expected,
    )
    assert d["positional_mask_bits"] == mask, (pattern, "masks", d["positional_mask_bits"], mask)
    assert d["contribution_bits"] == online + extra_online, (
        pattern,
        "DC online",
        d["contribution_bits"],
        online + extra_online,
    )
    assert d["shuffle_table_bits"] == shuffle
    assert d["logical_rounds"] == rounds + 2 * alert, (
        pattern,
        "rounds",
        d["logical_rounds"],
        rounds + 2 * alert,
    )
