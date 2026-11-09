from bisect import bisect_left, bisect_right
from collections import defaultdict


def degrees(xs, ys, lower, upper, p, q):
    amounts = sorted({x[1] for x in xs})
    tree = [0] * (len(amounts) + 1)
    events = []
    for j, y in enumerate(ys):
        events.extend(((y[0] - lower, 1, j), (y[0] - upper - 1, -1, j)))
    answer = [0] * len(ys)
    points = sorted(xs)
    index = 0

    def prefix(i):
        value = 0
        while i:
            value += tree[i]
            i -= i & -i
        return value

    for t, sign, j in sorted(events):
        while index < len(points) and points[index][0] <= t:
            pos = bisect_left(amounts, points[index][1]) + 1
            while pos < len(tree):
                tree[pos] += 1
                pos += pos & -pos
            index += 1
        low = (q * ys[j][1] + q + p - 1) // (q + p)
        high = q * ys[j][1] // (q - p)
        if low <= high:
            answer[j] += sign * (
                prefix(bisect_right(amounts, high)) - prefix(bisect_left(amounts, low))
            )
    return answer


def aggregate(incoming, outgoing, c, lower, upper):
    pattern = c.get("pattern", "P1")
    p, q = c.get("p", 5), c.get("q", 100)
    if pattern == "P3":
        times = sorted(
            x[0] for x in incoming if c.get("a_lo", 1) <= x[1] < c.get("a_hi", 1 << c.get("ba", 40))
        )
        return sum(
            bisect_right(times, t) - bisect_left(times, t - upper) - 1 >= c.get("m", 2) - 1
            for t in times
        )
    if pattern == "P4":
        xs, ys = defaultdict(list), defaultdict(list)
        for row in outgoing:
            xs[row[2]].append(row)
        for row in incoming:
            ys[row[2]].append(row)
        return sum(sum(degrees(xs[key], values, lower, upper, p, q)) for key, values in ys.items())
    d = degrees(incoming, outgoing, lower, upper, p, q)
    return sum(y[1] for y, count in zip(outgoing, d) if count) if pattern == "P2" else sum(d)
