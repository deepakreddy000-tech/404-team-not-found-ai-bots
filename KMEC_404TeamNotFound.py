"""KMEC Champions League 2026 – offline, single-file adaptive strategy bot.

Official get_bids signature. Whitelisted imports only; zero files, network, or APIs.
The strategy combines recent-opponent modeling, discrete expected utility,
scenario-correlated sweep estimates, controlled randomization, energy pacing.
"""
import math
import random


def _alloc(weights, budget):
    if budget <= 0:
        return (0, 0, 0)
    total = weights[0] + weights[1] + weights[2]
    if total <= 0:
        return (0, 0, 0)
    a = [int(budget * w / total) for w in weights]
    remainders = [(budget * weights[i] / total - a[i], i) for i in range(3)]
    remainders.sort(reverse=True)
    for j in range(budget - sum(a)):
        a[remainders[j][1]] += 1
    return (a[0], a[1], a[2])


def _profiles(values):
    # Templates have very different concentration; the values change every round.
    rank = sorted(range(3), key=lambda i: (values[i], i), reverse=True)
    balanced = [1.0, 1.0, 1.0]
    val = [max(0.2, float(x)) for x in values]
    squared = [x * x for x in val]
    top2 = [0.0, 0.0, 0.0]
    top2[rank[0]] = 0.60
    top2[rank[1]] = 0.40
    focus = [0.025, 0.025, 0.025]
    focus[rank[0]] = 1.0
    return (balanced, val, squared, top2, focus)


def _history_model(history, opp_energy):
    # Opponent spending statistics plus posterior over 5 behavioral templates.
    # Smoothing keeps robust alternatives alive against a deceptive opponent.
    weights = [0.23, 0.23, 0.15, 0.24, 0.15]
    totals = []
    observations = []
    for h in history[-11:]:
        if not isinstance(h, dict):
            continue
        try:
            bids = h['opp_bids']
            vals = h['lane_values']
            if len(bids) != 3 or len(vals) != 3:
                continue
            b = [max(0, int(x)) for x in bids]
            v = [max(0, int(x)) for x in vals]
            spent = sum(b)
            totals.append(spent)
            if spent:
                observations.append((b, v, spent))
        except (TypeError, ValueError, KeyError, OverflowError):
            continue
    # Recency-weighted likelihood; each model gets a modest uniform floor.
    if observations:
        logw = [math.log(w) for w in weights]
        for b, v, spent in observations[-8:]:
            templates = _profiles(v)
            for j, w in enumerate(templates):
                predicted = _alloc(w, spent)
                dist = sum(abs(b[i] - predicted[i]) for i in range(3)) / max(5.0, spent)
                logw[j] -= 1.3 * min(3.0, dist)
        maximum = max(logw)
        w = [math.exp((x-maximum) * 0.80) for x in logw]
        summ = sum(w)
        weights = [0.80 * x / summ + 0.04 for x in w]
    if totals:
        recent = totals[-5:]
        avg = sum(recent) / len(recent)
        dev = sum(abs(t - avg) for t in recent) / len(recent)
    else:
        avg, dev = 18.0, 6.0
    return weights, avg, dev, totals


def _scenario_bids(values, enemy_energy, round_num, history, rng, count=84):
    weights, avg, dev, totals = _history_model(history, enemy_energy)
    cum = []
    s = 0.0
    for v in weights:
        s += v
        cum.append(s)
    scenarios = []
    patterns = _profiles(values)
    empirical = []
    for h in history[-9:]:
        try:
            hb = tuple(max(0, int(x)) for x in h['opp_bids'])
            hv = tuple(max(0, int(x)) for x in h['lane_values'])
            if len(hb) == 3 and len(hv) == 3:
                s0 = sum(hv) or 1
                s1 = sum(values) or 1
                proximity = sum(abs(hv[i] / s0 - values[i] / s1) for i in range(3))
                empirical.append((proximity, hb))
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    empirical.sort(key=lambda x: x[0])
    empirical = empirical[:6]
    for j in range(count):
        r = rng.random()
        if empirical and r < 0.38:
            # Exact observed lane preferences capture opponents that repeatedly
            # favor A/B/C regardless of lane value (unmodelled by Blotto priors).
            base = rng.choice(empirical[:min(4, len(empirical))])[1]
            b = list(base)
            if rng.random() < 0.38:
                b = [max(0, x + rng.choice((-2, -1, 0, 0, 1, 2))) for x in b]
            budget = sum(b)
            if budget > enemy_energy:
                scenarios.append(_alloc(b, enemy_energy))
            else:
                scenarios.append(tuple(b))
            continue
        if history and r < 0.67:
            # Use empirical expenditure distribution with noise. Recognizes
            # recurring 18-bid templates and sudden changes in intensity.
            budget = rng.choice(totals[-6:]) if totals else avg
            budget += rng.gauss(0, max(1.2, dev * 0.45))
        elif r < 0.94:
            budget = rng.choice((8, 11, 14, 16, 18, 20, 23, 26, 30, 36, 45))
        else:
            budget = rng.uniform(0, max(20, enemy_energy))
        if round_num >= 13 and rng.random() < 0.20:
            budget = rng.uniform(max(0.0, 0.65 * enemy_energy), enemy_energy)
        budget = min(enemy_energy, max(0, int(round(budget))))
        p = rng.random()
        if p < 0.82:
            u = rng.random()
            k = 0
            while k < 4 and u > cum[k]:
                k += 1
            base = patterns[k]
            if rng.random() < 0.40:
                alloc = _alloc(base, budget)
            else:
                noisy = [max(0.0, v * rng.uniform(0.68, 1.32)) for v in base]
                alloc = _alloc(noisy, budget)
        else:
            # Unknown and adversarial bidders. Rare but intentionally retained.
            rand = [rng.expovariate(1.0) for _ in range(3)]
            alloc = _alloc(rand, budget)
        scenarios.append(alloc)
    return scenarios


def _lane_options(samples, cap, lane):
    # Values that just beat frequent predicted opponent bids have a large
    # expected-value jump per single extra energy spent.
    freq = {}
    for s in samples:
        x = s[lane]
        if x < cap:
            freq[x + 1] = freq.get(x + 1, 0) + 1
    top = sorted(freq, key=lambda x: freq[x], reverse=True)[:11]
    # At most 16 choices per lane, so the cubic combination search stays fast
    # even if 200+ energy is carried into the final round.
    opts = set([0, 1, 2, 3, cap])
    opts.update(top)
    return sorted(x for x in opts if 0 <= x <= cap)


def _tables(samples, cap):
    n = float(len(samples))
    wi = []
    lo = []
    for i in range(3):
        f = [0] * (cap + 1)
        greater = 0
        for s in samples:
            x = s[i]
            if x <= cap:
                f[x] += 1
            else:
                greater += 1
        cumulative = 0
        win = []
        lose = []
        for b in range(cap + 1):
            # Strictly higher bid wins, exactly equal earns 0.
            win.append(cumulative / n)
            lose.append((len(samples) - cumulative - f[b]) / n)
            cumulative += f[b]
        wi.append(win)
        lo.append(lose)
    return wi, lo


def get_bids(round_num: int, my_energy: int, opp_energy: int,
             lane_values: list[int], history: list[dict]) -> tuple[int, int, int]:
    """Return 3 legal integer bids; deterministic rules, randomized mixed policy."""
    try:
        me = max(0, int(my_energy))
        enemy = max(0, int(opp_energy))
        rnd = max(1, min(15, int(round_num)))
        values = tuple(max(0, int(x)) for x in lane_values)
        if len(values) != 3:
            return (0, 0, 0)
    except (TypeError, ValueError, OverflowError):
        return (0, 0, 0)
    if me == 0 or max(values) == 0:
        return (0, 0, 0)
    if enemy == 0:
        result = [0, 0, 0]
        for i in sorted(range(3), key=lambda k: values[k], reverse=True):
            if me > 0 and values[i] > 0:
                result[i] = 1
                me -= 1
        return tuple(result)
    if not isinstance(history, (list, tuple)):
        history = []

    rng = random.Random(random.getrandbits(56))
    scenarios = _scenario_bids(values, enemy, rnd, history, rng)
    rounds_left = 15 - rnd
    sustainable = (me + 10 * rounds_left) / (rounds_left + 1)
    score_me = history[-1].get('my_score', 0) if history and isinstance(history[-1], dict) else 0
    score_enemy = history[-1].get('opp_score', 0) if history and isinstance(history[-1], dict) else 0
    score_gap = score_me - score_enemy
    importance = max(.80, min(1.25, sum(values) / 17.0))
    # Slack early is intentional: enough flexibility for high-value rounds,
    # but do not spend the entire starting bank on a single round.
    multiplier = (1.58 if rnd < 7 else 1.72 if rnd < 12 else 2.10)
    if score_gap < -18:
        multiplier *= 1.12
    if score_gap > 30 and rnd >= 10:
        multiplier *= 0.82
    cap = min(me, max(3, int(sustainable * multiplier * importance)))
    if rnd <= 4 and sum(values) >= 21:
        cap = min(me, max(cap, int(sustainable * 2.45)))
    if rnd == 15:
        cap = me
    elif rnd == 14:
        cap = min(me, max(cap, int(sustainable * 3.0)))
    # A player with fewer current reserves shouldn't necessarily overbid.
    cap = min(cap, enemy + 3) if enemy < 3 and rnd < 15 else cap

    win, lose = _tables(scenarios, cap)
    choices = [_lane_options(scenarios, cap, i) for i in range(3)]
    future_weight = rounds_left / 14.0
    price = 0.82 * (0.12 + 1.22 * future_weight)
    if me < 13 and rounds_left > 3:
        price *= 1.30
    if rnd >= 13:
        price *= 0.55
    if rnd == 15:
        price = 0.012
    if score_gap > 35 and rnd >= 12:
        price *= 1.35
    if score_gap < -35 and rnd >= 12:
        price *= 0.75

    candidates = []
    v0, v1, v2 = values
    w0, w1, w2 = win
    l0, l1, l2 = lose
    for a in choices[0]:
        for b in choices[1]:
            ab = a + b
            if ab > cap:
                continue
            for c in choices[2]:
                spent = ab + c
                if spent > cap:
                    continue
                # Exact marginal values and approximate joint sweep odds.
                score = (v0 * (w0[a] - l0[a])
                         + v1 * (w1[b] - l1[b])
                         + v2 * (w2[c] - l2[c])
                         + 5.0 * (w0[a] * w1[b] * w2[c]
                                  - 1.3 * l0[a] * l1[b] * l2[c])
                         - price * spent)
                candidates.append((score, (a, b, c)))
    if not candidates:
        return (0, 0, 0)
    candidates.sort(reverse=True)
    # Re-score most promising 15 candidates on full *joint* opponent scenarios.
    best = (0, 0, 0)
    best_score = -1e100
    top = candidates[:15]
    n = float(len(scenarios))
    for prelim, action in top:
        a, b, c = action
        total = 0.0
        tail = []
        for x, y, z in scenarios:
            p = ((v0 if a > x else -v0 if a < x else 0)
                 + (v1 if b > y else -v1 if b < y else 0)
                 + (v2 if c > z else -v2 if c < z else 0))
            if a > x and b > y and c > z:
                p += 5
            if a < x and b < y and c < z:
                p -= 6.5
            total += p
            tail.append(p)
        tail.sort()
        k = max(1, len(tail) // 5)
        tail_avg = sum(tail[:k]) / k
        # A small worst-case component protects against inaccurate models.
        score = 0.75 * (total / n) + 0.25 * tail_avg - price * (a + b + c)
        # Small mixing noise only among similarly strong candidate actions.
        score += rng.uniform(-0.16, 0.16)
        if score > best_score:
            best_score = score
            best = action
    return (int(best[0]), int(best[1]), int(best[2]))
