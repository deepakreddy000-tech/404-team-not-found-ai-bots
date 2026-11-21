"""404 Team Not Found — V6 FINAL Adaptive Multi-Front Commander.

Self-contained offline entry for the Node War game engine. Preserves exact
combat resolution, observed enemy-composition learning, and fog-aware defense.
Conditionally shifts recruitment to a southern pressure plan based on the
observable central expedition size and its reinforcement schedule.
"""
from time import perf_counter
from random import random

from nodewar import BotBase

KINDS = ("raider", "guard", "siege", "scout")
COST = (10, 12, 18, 6)
HP = (20, 30, 35, 10)
EMPTY = (0, 0, 0, 0)
NEUTRAL = {4: (1, 0, 0, 0), 7: (0, 2, 0, 0), 10: (0, 2, 1, 0)}
MIRROR = (14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)


def army(units):
    return tuple(units.get(k, 0) for k in KINDS) if units else EMPTY


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])


def fighters(a):
    return a[0] + a[1] + a[2]


def value(a):
    return 10 * a[0] + 12 * a[1] + 18 * a[2] + 6 * a[3]


def with_units(a, kind, count):
    result = list(a)
    result[kind] += count
    return tuple(result)


def damage(a, b):
    """Engine-equivalent per-type damage, preserving floating-point operation order."""
    total = sum(b)
    if not total:
        return 0.0
    result = 0.0
    for k in range(4):
        count = a[k]
        if count <= 0:
            continue
        if k == 3:  # Scouts cannot attack.
            continue
        counter = (2, 0, 1)[k]
        weak = (1, 2, 0)[k]
        positive = b[counter]
        negative = b[weak]
        neutral = total - positive - negative
        multiplier = (2.0 * positive + 1.0 * neutral + 0.5 * negative) / total
        result += count * (12, 14, 22)[k] * multiplier
    return result


def casualties(a, dmg):
    """Exactly match nodewar.apply_damage_to_army ordering and float rounding."""
    effective = dmg
    if effective <= 0 or not sum(a):
        return a
    hp = sum(a[k] * HP[k] for k in range(4))
    if effective >= hp:
        return EMPTY
    ratio = effective / hp
    survivors = list(a)
    carried = 0.0
    for k in (0, 2, 1, 3):
        count = survivors[k]
        if count <= 0:
            continue
        hp_per = HP[k]
        type_damage = (count * hp_per) * ratio + carried
        lost = int(type_damage // hp_per)
        rem = type_damage % hp_per
        if rem >= (hp_per * 0.5) and lost < count:
            lost += 1
            carried = 0.0
        else:
            carried = rem
        survivors[k] = count - min(count, lost)
    return tuple(survivors)


_BATTLE_CACHE = {}

def battle(d, a, reduction):
    """Forecast battle outcomes using the published Node War combat resolver."""
    cache_key = (d, a, reduction)
    previous = _BATTLE_CACHE.get(cache_key)
    if previous is not None:
        return previous
    d_mod = 1.15 if d[3] else (0.85 if a[3] else 1.0)
    a_mod = 1.15 if a[3] else (0.85 if d[3] else 1.0)
    for _ in range(10):
        if not fighters(d) or not fighters(a):
            break
        dd, ad = damage(d, a) * d_mod, damage(a, d) * a_mod
        next_d = casualties(d, ad * (1 - reduction))
        next_a = casualties(a, dd)
        if next_d == d and next_a == a:
            if dd >= ad:
                k = next(k for k in (0, 2, 1) if a[k])
                next_a = with_units(a, k, -1)
            else:
                k = next(k for k in (0, 2, 1) if d[k])
                next_d = with_units(d, k, -1)
        d, a = next_d, next_a
    if not fighters(d) and fighters(a):
        d = EMPTY
    if not fighters(a) and fighters(d):
        a = EMPTY
    outcome = (d, a)
    if len(_BATTLE_CACHE) >= 8192:
        _BATTLE_CACHE.clear()
    _BATTLE_CACHE[cache_key] = outcome
    return outcome


class CompetitorBot(BotBase):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.home = 0 if player_id == 1 else 14
        self.enemy = 3 - player_id
        self.enemy_home = 14 - self.home
        self.adj = None
        self.sightings = {}
        # Dirichlet-style prior over enemy combat-unit proportions, updated
        # from unique, *revealed* fleets and gently discounted over time.
        self.enemy_mix = [2.0, 2.0, 2.0]
        self.enemy_training_rate = 0.0
        self.enemy_training_samples = 0
        self.last_enemy_garrisons = {}
        self.previous_enemy_arrivals = set()
        self.seen_fleets = {}
        self.northern_raid_ready = False
        self.northern_counter_ready = False
        self.northern_siege_hold = 0
        self.south_assault_quota = 0
        # Mix indistinguishable early-game options to reduce exploitability.
        self.south_early_style = 5 if random() < (2.0 / 3.0) else 6

    def _setup(self, obs):
        self.adj = {i: {} for i in obs.nodes}
        for lane in obs.lanes:
            i, j, turns = lane["node_a"], lane["node_b"], lane["travel_turns"]
            self.adj[i][j] = turns
            self.adj[j][i] = turns
        self.rank = {i: i if self.home == 0 else MIRROR[i] for i in obs.nodes}

    def _opening(self, t, owned, stock, enemy_arrivals, budget, send, train):
        if t > 4 or self.home not in owned or self.home in enemy_arrivals:
            return
        n, s, gate = ((1, 2, 3) if self.home == 0 else (13, 12, 11))
        if t == 1 and budget >= 42 and stock[self.home][1] and stock[self.home][3]:
            send(self.home, n, (0, 1, 0, 0))
            send(self.home, s, (1, 0, 0, 1))
            train(self.home, 2, 2)
            train(self.home, 3, 1)
        elif t == 2 and stock[self.home][2] >= 2:
            send(self.home, gate, (0, 0, 2, 0))
            train(self.home, 0, 1)
            train(self.home, 3, 1)
        elif t == 3 and stock[self.home][0] and stock[self.home][3]:
            send(self.home, gate, (1, 0, 0, 1))
        elif t == 4:
            for bastion in (n, s):
                if bastion in owned and bastion not in enemy_arrivals:
                    train(bastion, 2, 1)

    def act(self, obs):
        deadline = perf_counter() + 0.025
        t = obs.current_turn
        if self.adj is None:
            self._setup(obs)
        if t == 1:
            self.sightings.clear()
            self.seen_fleets.clear()
            self.enemy_mix = [2.0, 2.0, 2.0]
            self.enemy_training_rate = 0.0
            self.enemy_training_samples = 0
            self.last_enemy_garrisons.clear()
            self.previous_enemy_arrivals.clear()
            self.northern_raid_ready = False
            self.northern_counter_ready = False
            self.northern_siege_hold = 0
            self.south_assault_quota = 0
        pid = self.player_id
        nodes = obs.nodes
        relic = obs.relic_node_id
        owned = {i for i in nodes if nodes[i].owner == pid}
        usable = {i for i in owned if not nodes[i].capturing_player}
        if not usable:
            return []
        ordered = sorted(usable, key=self.rank.get)
        stock = {i: army(nodes[i].units) for i in usable}
        training = dict.fromkeys(usable, EMPTY)
        hostile = {i for i in nodes if nodes[i].owner == self.enemy
                   or nodes[i].capturing_player == self.enemy}
        income = sum(nodes[i].production for i in owned)
        free = 10 + 4 * len(owned)
        domestic = sum(map(sum, stock.values()))
        foreign = sum(f.total_count for f in obs.my_fleets)
        foreign += sum(sum(army(n.units)) for n in nodes.values() if n.capturing_player == pid)
        upkeep = max(0, domestic - free) + 2 * max(0, foreign - max(0, free - domestic))
        budget = max(0, obs.my_credits + income - upkeep)
        population = domestic + foreign
        orders = []
        friendly, hostile_fleets = {}, {}
        incoming_friendly, incoming_hostile = {}, {}

        def note(table, key, force):
            table[key] = add(table.get(key, EMPTY), force)

        def send(i, j, force):
            stock[i] = subtract(stock[i], force)
            eta = self.adj[i][j]
            note(friendly, (j, eta), force)
            note(incoming_friendly, j, force)
            orders.append(self.move(i, j, {k: n for k, n in zip(KINDS, force) if n}))

        def train(i, kind, count):
            nonlocal budget, population
            count = min(count, budget // COST[kind])
            if count:
                orders.append(self.train(i, KINDS[kind], count))
                training[i] = with_units(training[i], kind, count)
                budget -= COST[kind] * count
                population += count

        # Forget stale evidence slowly; later revealed behavior matters more.
        if t > 1:
            self.enemy_mix = [1.5 + (x - 1.5) * 0.985 for x in self.enemy_mix]
        observed_fleets = {}
        for f in obs.enemy_fleets:
            if f.units is not None:
                force = army(f.units)
                observed_fleets[f.id] = force
                if f.id not in self.seen_fleets:
                    for k, count in enumerate(force[:3]):
                        self.enemy_mix[k] += count
            elif f.id in self.seen_fleets:
                observed_fleets[f.id] = self.seen_fleets[f.id]
        self.seen_fleets = observed_fleets
        for i in hostile:
            if nodes[i].units is not None and nodes[i].capturing_player != pid:
                force_seen = army(nodes[i].units)
                before = self.last_enemy_garrisons.get(i)
                # Adjacent observations on successive turns permit noisy
                # recruitment-rate inference. Do not treat arriving fleets as
                # recruitment, and cap large unexplained changes.
                if before and before[0] == t - 1 and i not in self.previous_enemy_arrivals:
                    growth = min(5, sum(max(0, force_seen[k] - before[1][k]) for k in range(4)))
                    self.enemy_training_rate = 0.84 * self.enemy_training_rate + 0.16 * growth
                    self.enemy_training_samples += 1
                self.last_enemy_garrisons[i] = (t, force_seen)
                self.sightings[i] = (t, force_seen)
        self.previous_enemy_arrivals = {f.dest_node for f in obs.enemy_fleets if f.turns_remaining == 1}
        current_mix = self.enemy_mix.copy()
        for i in hostile:
            if nodes[i].capturing_player == pid:
                continue
            for k, count in enumerate(army(nodes[i].units)[:3]):
                current_mix[k] += 1.5 * min(25, count)
        majority = max(range(3), key=current_mix.__getitem__)

        for f in obs.enemy_fleets:
            if f.id not in observed_fleets:
                origin = self.sightings.get(f.origin_node, (0, EMPTY))[1]
                # Blend global Dirichlet evidence with the last visible army
                # at the departure node. Never invent a known composition.
                priors = [max(0.25, x) for x in current_mix]
                if fighters(origin):
                    priors = [0.55 * priors[k] + 0.45 * (origin[k] + 1) *
                              (sum(priors) / (fighters(origin) + 3)) for k in range(3)]
                scout_guess = int(f.total_count >= 4 and (origin[3] > 0 or f.total_count >= 9))
                n_combat = f.total_count - scout_guess
                normalizer = sum(priors)
                scaled = [n_combat * p / normalizer for p in priors]
                concealed = [int(x) for x in scaled]
                for k in sorted(range(3), key=lambda k: (scaled[k] - concealed[k], priors[k]), reverse=True)[:n_combat - sum(concealed)]:
                    concealed[k] += 1
                force = (*concealed, scout_guess)
            else:
                force = observed_fleets[f.id]
            note(hostile_fleets, (f.dest_node, f.turns_remaining), force)
            note(incoming_hostile, f.dest_node, force)
        for f in obs.my_fleets:
            force = army(f.units)
            note(friendly, (f.dest_node, f.turns_remaining), force)
            note(incoming_friendly, f.dest_node, force)

        self._opening(t, usable, stock, incoming_hostile, budget, send, train)
        if t <= 3 and len(orders) >= 1 and not any(i in incoming_hostile for i in usable):
            return orders

        def importance(i):
            return 15 if i == self.home else (13 if i == relic else nodes[i].production)


        gate = 3 if self.home == 0 else 11
        rival_gate = MIRROR[gate]
        reinforce_opening = (
            t == 6 and gate in usable and gate not in incoming_hostile
            and nodes[relic].owner == 0
            and (nodes[rival_gate].owner == self.enemy
                 or nodes[rival_gate].capturing_player == self.enemy)
        )
        if reinforce_opening:
            # A fast enemy can reach the Relic on turn 8. Let the first fight
            # resolve, then arrive with a larger force while capture is pending.
            for kind, count in ((0, 3), (2, 1), (3, 1)):
                train(gate, kind, count)

        # Anti-mirror flank: an opponent reinforcing the central Relic from both
        # adjacent outposts can leave its northern bridge castle underdefended.
        # Trigger only on *visible* enemy support fleets. Unlike a hard-coded
        # flank rush, this preserves our central strategy versus other opponents.
        north = 4 if self.home == 0 else 10
        other_north = 10 if self.home == 0 else 4
        enemy_shoulder = (8, 9) if self.home == 0 else (5, 6)
        if t == 9 and north in usable and nodes[other_north].owner == self.enemy:
            near_relic_support = sum(
                1 for f in obs.enemy_fleets
                if (f.dest_node == relic and f.origin_node in enemy_shoulder
                    and f.turns_remaining == 1 and f.total_count > 0)
            )
            visible_defense = nodes[other_north].units
            if (near_relic_support >= 2 and visible_defense is not None
                    and fighters(army(visible_defense)) <= 2 and budget >= 12):
                train(north, 1, 1)
                # Extra Siege is only justified against the distinctive
                # 10-unit central expedition used by the mirrored heavy build.
                # Smaller expeditions (e.g. 8-unit Siege reinforcement) need
                # central defense more than an expensive speculative flank.
                central_size = max((f.total_count for f in obs.enemy_fleets
                                    if f.origin_node == rival_gate and f.dest_node == relic
                                    and f.turns_remaining == 1), default=0)
                if central_size == 10 and budget >= 18:
                    train(north, 2, 1)
                self.northern_raid_ready = True
        elif t == 10 and self.northern_raid_ready:
            if (north in usable and nodes[other_north].owner == self.enemy
                    and not incoming_hostile.get(north)):
                defenders = nodes[other_north].units
                force = stock[north]
                if (defenders is not None and fighters(army(defenders)) <= 2
                        and fighters(force) >= 2):
                    if army(defenders)[2] and budget >= 10:
                        # An enemy Siege build threatens the bridge; return our
                        # speculative extra Siege to the central front while Guards hold.
                        center = 5 if self.home == 0 else 9
                        if stock[north][2] and center in owned:
                            send(north, center, (0, 0, 1, 0))
                        train(north, 0, 1)
                        self.northern_siege_hold = t + 2
                    else:
                        send(north, other_north, force)
                elif (defenders is not None and army(defenders)[1] >= 3
                        and fighters(force) >= 2 and budget >= 18):
                    train(north, 2, 1)
                    self.northern_counter_ready = True
            self.northern_raid_ready = False

        # Conditional multi-front pressure. The central enemy fleet is already
        # visible on radar: a smaller (<=9) attack is a reason to conserve cash
        # for Relic defense; a large expedition leaves its southern wing exposed.
        # Dispatch a small detachment while retaining a garrison at the source.
        south = 6 if self.home == 0 else 8
        opposite_south = 8 if self.home == 0 else 6
        southern_sortie = False
        if t == 8 and south in usable and nodes[opposite_south].owner == self.enemy:
            central_size = max((f.total_count for f in obs.enemy_fleets
                                if f.origin_node == rival_gate and f.dest_node == relic
                                and f.turns_remaining == 2), default=0)
            self.south_assault_quota = self.south_early_style if central_size >= 10 else 4
            if south not in incoming_hostile:
                train(south, 0, self.south_assault_quota)
                southern_sortie = True
        if t == 9 and self.south_assault_quota and south in usable and nodes[opposite_south].owner == self.enemy:
            if not incoming_hostile.get(south) and stock[south][0] >= self.south_assault_quota:
                central_size = max((f.total_count for f in obs.enemy_fleets
                                    if f.origin_node == rival_gate and f.dest_node == relic
                                    and f.turns_remaining == 1), default=0)
                support_to_relic = sum(1 for f in obs.enemy_fleets
                                       if (f.dest_node == relic and f.origin_node in enemy_shoulder
                                           and f.turns_remaining == 1))
                # With smaller central fleets or double-support at the Relic,
                # preserve one more Raider and additional Gold for later defense.
                seen_scout = army(nodes[opposite_south].units)[3] > 0
                count = 3 if seen_scout else (3 if central_size < 10 or support_to_relic >= 2 else 4)
                send(south, opposite_south, (min(stock[south][0] - 1, count), 0, 0, 0))
                southern_sortie = True
            self.south_assault_quota = 0

        # Reinforcements shown as arriving this turn join the defense immediately.
        # Keep enough mobile units in place, buying the cheapest viable counter force.
        urgent = sorted(
            (i for i in usable if i in incoming_hostile),
            key=lambda i: (not bool(sum(hostile_fleets.get((i, 1), EMPTY))), -importance(i), self.rank[i]),
        )
        frozen = set()
        if southern_sortie:
            frozen.add(south)
        for i in urgent:
            if perf_counter() >= deadline:
                return orders
            threat = hostile_fleets.get((i, 1), EMPTY)
            if not fighters(threat):
                if fighters(incoming_hostile[i]):
                    frozen.add(i)
                continue
            frozen.add(i)
            defense = add(add(stock[i], training[i]), friendly.get((i, 1), EMPTY))
            bonus = nodes[i].defense_bonus
            best = None
            for scout in range(1 + int(not defense[3] and budget >= 6)):
                start = with_units(defense, 3, scout)
                for kind in range(3):
                    cash = budget - 6 * scout
                    upper = min(cash // COST[kind], sum(threat) * 2 + 4)
                    lo, hi = 0, upper + 1
                    while lo < hi:
                        middle = (lo + hi) // 2
                        candidate = with_units(start, kind, middle)
                        survivors, _ = battle(candidate, threat, bonus)
                        if fighters(survivors):
                            hi = middle
                        else:
                            lo = middle + 1
                    for count in range(lo, min(upper, lo + 2) + 1):
                        candidate = with_units(start, kind, count)
                        survivors, _ = battle(candidate, threat, bonus)
                        if fighters(survivors):
                            cost = COST[kind] * count + 6 * scout
                            loss = value(candidate) - value(survivors)
                            choice = (loss + 0.32 * cost, cost, -value(survivors), kind, count, scout)
                            if best is None or choice < best:
                                best = choice
            mixed_best = None
            # A pure counter-unit purchase is not necessarily optimal against
            # mixed attackers. Search small affordable two-type garrisons with
            # the exact battle resolver, only when danger is imminent.
            if best is not None and fighters(threat) >= 4 and budget >= 24 and not any(
                    f.dest_node == i and f.turns_remaining == 1 and f.units is None
                    and f.id not in observed_fleets for f in obs.enemy_fleets):
                for first_kind, second_kind in ((0, 1), (0, 2), (1, 2)):
                    if perf_counter() >= deadline - 0.009:
                        break
                    for scout_count in (0, 1):
                        for a_count in range(min(6, budget // COST[first_kind]) + 1):
                            remaining_cash = budget - COST[first_kind] * a_count - 6 * scout_count
                            if remaining_cash < 0:
                                continue
                            for b_count in range(min(6, remaining_cash // COST[second_kind]) + 1):
                                if not a_count or not b_count:
                                    continue
                                if perf_counter() >= deadline - 0.007:
                                    break
                                proposed = with_units(with_units(with_units(defense, first_kind, a_count),
                                                                 second_kind, b_count), 3, scout_count)
                                survivors, _ = battle(proposed, threat, bonus)
                                if not fighters(survivors):
                                    continue
                                price = COST[first_kind] * a_count + COST[second_kind] * b_count + 6 * scout_count
                                loss = value(proposed) - value(survivors)
                                score = (loss + 0.32 * price, price, -value(survivors))
                                if score < best[:3] and (mixed_best is None or score < mixed_best[0]):
                                    mixed_best = (score, first_kind, a_count, second_kind, b_count, scout_count)
            if best is not None:
                _, _, _, kind, count, scout = best
                # For a concealed fleet, evaluate three plausible counter armies,
                # rather than assuming its full roster matches one predicted mix.
                concealed = [f for f in obs.enemy_fleets
                             if f.dest_node == i and f.turns_remaining == 1
                             and f.units is None and f.id not in observed_fleets]
                if concealed and budget >= 12 and importance(i) >= 7:
                    hidden_count = sum(f.total_count for f in concealed)
                    # Subtract the known estimation of the hidden component.
                    h_mix = [max(0.1, x) for x in self.enemy_mix]
                    h_total = sum(h_mix)
                    probabilities = [v / h_total for v in h_mix]
                    scout_estimate = int(hidden_count >= 5)
                    known = EMPTY
                    for fleet in obs.enemy_fleets:
                        if (fleet.dest_node == i and fleet.turns_remaining == 1
                                and fleet.id in observed_fleets):
                            known = add(known, observed_fleets[fleet.id])
                    trials = [add(known, with_units(EMPTY, k, hidden_count)) for k in range(3)]
                    # No full expensive search when a revealed threat is already handled.
                    contenders = []
                    for scout_count in (0, 1):
                        if 6 * scout_count > budget:
                            continue
                        for unit_kind in range(3):
                            limit = min(12, (budget - 6 * scout_count) // COST[unit_kind],
                                        max(3, hidden_count + 3))
                            for ct in range(limit + 1):
                                if ct == 0 and scout_count == 0:
                                    continue
                                if perf_counter() >= deadline - 0.003:
                                    break
                                proposal = with_units(with_units(defense, unit_kind, ct), 3, scout_count)
                                chance = sum(probabilities[k] * bool(fighters(battle(proposal, trials[k], bonus)[0]))
                                             for k in range(3))
                                price = COST[unit_kind] * ct + 6 * scout_count
                                contenders.append((chance, price, unit_kind, ct, scout_count))
                    if contenders:
                        threshold = 0.82 if i in (self.home, relic) else 0.68
                        acceptable = [p for p in contenders if p[0] >= threshold]
                        if acceptable:
                            win_chance, price, kind, count, scout = min(
                                acceptable, key=lambda p: (p[1], -p[0]))
                        # Else preserve baseline defence against the model's
                        # highest-probability composition.
                if mixed_best is not None:
                    _, first_kind, a_count, second_kind, b_count, scout = mixed_best
                    train(i, first_kind, a_count)
                    train(i, second_kind, b_count)
                else:
                    train(i, kind, count)
                train(i, 3, scout)

        # Anti-mirror tactical response: two-turn support from freshly captured
        # outposts synchronizes with our three-turn gate fleet at the Relic.
        # Only commit if a visible opposing fleet arrives at the same moment.
        if (t >= 6 and relic not in owned and
                fighters(friendly.get((relic, 2), EMPTY)) >= 5 and
                fighters(hostile_fleets.get((relic, 2), EMPTY)) >= 5):
            support = []
            for staging in self.adj[relic]:
                if (staging in usable and staging not in frozen and
                        self.adj[relic][staging] == 2):
                    f = stock[staging]
                    if fighters(f) and not incoming_hostile.get(staging):
                        support.append((staging, f))
            for staging, f in support:
                if fighters(f):
                    send(staging, relic, f)

        # Defend the bridge with a Siege unit and pressure the now-vacated
        # enemy end with the original two Guards if a large enemy fleet leaves.
        if t == 11 and self.northern_counter_ready:
            counterfleet = any(f.origin_node == other_north and f.dest_node == north
                               and f.turns_remaining == 2 and f.total_count >= 3
                               for f in obs.enemy_fleets)
            defenders = nodes[other_north].units
            if (counterfleet and north in usable and north not in frozen
                    and defenders is not None and fighters(army(defenders)) == 0
                    and stock[north][1] >= 2 and stock[north][2] >= 1):
                send(north, other_north, (0, 2, 0, 0))
                frozen.add(north)
            self.northern_counter_ready = False

        if t <= self.northern_siege_hold and north in usable:
            frozen.add(north)

        available = {i: stock[i] for i in usable if i not in frozen}
        forbidden = {i for i in nodes if nodes[i].capturing_player == pid}
        remaining = obs.max_turns - t + 1
        score_delta = obs.my_score - obs.opponent_score
        enemy_node_count = sum(n.owner == self.enemy for n in nodes.values())
        projected_rate = len(owned) - enemy_node_count
        if obs.relic_active:
            projected_rate *= 2
            projected_rate += 10 * ((relic in owned) - (nodes[relic].owner == self.enemy))
        safe_lead = projected_rate > 0 and score_delta + (remaining - 3) * projected_rate > 10

        def target_value(i):
            n = nodes[i]
            if i == relic:
                return (33 if obs.relic_active or obs.relic_charging else 25) + (8 if not safe_lead else 0)
            if n.owner == 0:
                side = self.rank[i]
                return 15 + n.production + (4 if side <= 6 else 0)
            if i == self.enemy_home:
                return 19 + (18 if enemy_node_count <= 2 else 0)
            return 19 + n.production + (3 if not safe_lead else 0)

        def defender(i):
            n = nodes[i]
            if n.units is not None:
                return army(n.units)
            if i in hostile:
                last_turn, last = self.sightings.get(i, (0, EMPTY))
                if last_turn and t - last_turn <= 5:
                    return last
                return with_units(EMPTY, majority, 2 if i == self.enemy_home else 1)
            return NEUTRAL.get(n.production, EMPTY)

        def rival_arrivals(i, eta):
            result = EMPTY
            for arrival in range(1, eta + 1):
                result = add(result, hostile_fleets.get((i, arrival), EMPTY))
            return result

        def attack_result(i, eta, force, reinforce=True):
            n = nodes[i]
            existing = defender(i)
            incoming = rival_arrivals(i, eta)
            if i not in hostile:
                if fighters(incoming):
                    left, force = battle(incoming, force, 0.0)
                    if fighters(left):
                        return EMPTY
                after, survivors = battle(existing, force, n.defense_bonus)
                return survivors if not fighters(after) else EMPTY
            defense = add(existing, incoming)
            if reinforce and n.owner == self.enemy and not n.capturing_player:
                # The opponent can train after a fleet appears on radar.
                extra = 1 if t < 12 else (2 if i == self.enemy_home or i == relic else 1)
                if enemy_node_count >= 5 and t >= 25:
                    extra += 2
                if eta == 3:
                    extra += 1
                # Learn whether this opponent tends to reinforce visible nodes
                # much faster than the basic conservative default expects.
                if self.enemy_training_samples >= 4 and self.enemy_training_rate >= 1.9:
                    extra += 1 if self.enemy_training_rate < 3.5 else 2
                counter = self._counter(force)
                defense = with_units(defense, counter, extra)
                if fighters(defense) >= 4 and not defense[3]:
                    defense = with_units(defense, 3, 1)
            after, survivors = battle(defense, force, n.defense_bonus)
            return survivors if not fighters(after) else EMPTY

        def launchable(i, j):
            a = available.get(i, EMPTY)
            if not fighters(a):
                return EMPTY
            local_hostiles = [k for k in self.adj[i] if k in hostile and k != j]
            if i == relic:
                reserve = max(2 if local_hostiles or t >= 58 else 1,
                              int(fighters(a) * (0.60 if safe_lead else 0.40)))
            elif i == self.home:
                reserve = 3 if local_hostiles else int(not a[3])
            elif (local_hostiles or j not in owned) and t > 9:
                reserve = max(1, min(2, fighters(a) // 4))
            else:
                reserve = 0
            if reserve == 0 and (i != self.home or not a[3]):
                return a
            counts = list(a)
            for kind in (1, 2, 0):
                hold = min(counts[kind], reserve)
                counts[kind] -= hold
                reserve -= hold
            if a[3] and i in (self.home, relic):
                counts[3] -= 1
            return tuple(counts)

        def group_for(j, eta):
            return [(i, launchable(i, j)) for i in sorted(self.adj[j], key=self.rank.get)
                    if i in available and self.adj[j][i] == eta]

        targets = [
            i for i in nodes
            if remaining > 2 and (i not in forbidden or (i == relic and i in incoming_hostile))
            and i not in usable and (i not in owned or i in hostile)
        ]
        targets.sort(key=lambda i: (-target_value(i), self.rank[i]))

        # Group fleets with matching arrival times. Neutral captures use the smallest
        # reliable detachment, preserving the rest of the army for parallel expansion.
        for j in targets:
            if perf_counter() >= deadline:
                return orders
            if j == relic and reinforce_opening:
                continue
            underway = j in incoming_friendly
            best = None
            for eta in (2, 3):
                if eta >= remaining:
                    continue
                reinforcing = friendly.get((j, eta), EMPTY)
                relief = False
                if j == relic and j not in hostile and fighters(hostile_fleets.get((j, eta), EMPTY)):
                    earlier = friendly.get((j, eta - 1), EMPTY)
                    if nodes[j].capturing_player == pid or fighters(earlier):
                        # A second force can arrive one turn later and secure the capture.
                        relief = True
                if underway and not relief:
                    if not fighters(reinforcing):
                        continue
                    initial_survivors = attack_result(j, eta, reinforcing)
                    if fighters(initial_survivors) and (
                            j not in hostile or value(initial_survivors) >= 0.55 * value(reinforcing)):
                        continue
                group = [(i, a) for i, a in group_for(j, eta) if fighters(a)]
                if not group:
                    continue
                variants = [[(i, a)] for i, a in group]
                if len(group) > 1:
                    variants.append(group)
                simple_neutral = j not in hostile and j not in incoming_hostile and j != relic
                if simple_neutral:
                    for i, a in group:
                        for kind in range(3):
                            for count in range(1, min(3, a[kind]) + 1):
                                f = with_units(EMPTY, kind, count)
                                variants.append([(i, f)])
                                if a[3]:
                                    variants.append([(i, with_units(f, 3, 1))])
                for dispatch in variants:
                    force = reinforcing
                    for _, a in dispatch:
                        force = add(force, a)
                    if relief:
                        occupied = defender(j) if j in forbidden else EMPTY
                        if fighters(earlier) and j not in forbidden:
                            occupied = attack_result(j, eta - 1, earlier)
                        after, survivors = battle(hostile_fleets[j, eta], add(occupied, force), 0.0)
                        if fighters(after):
                            continue
                    else:
                        survivors = attack_result(j, eta, force)
                    if not fighters(survivors):
                        continue
                    loss = max(0, value(force) - value(survivors))
                    if j in hostile and safe_lead and j != relic and loss > 0.22 * value(force) + 14:
                        continue
                    if j in hostile and fighters(survivors) < max(1, fighters(force) // 5):
                        continue
                    if simple_neutral:
                        metric = -loss * 1.25 - 0.20 * value(force) - 9 * (eta - 2)
                    else:
                        metric = 0.33 * value(survivors) - loss - 13 * (eta - 2)
                    if underway:
                        metric -= 0.14 * (value(force) - value(reinforcing))
                    if best is None or metric > best[0]:
                        best = (metric, dispatch)
            if best:
                for i, force in best[1]:
                    send(i, j, force)
                    available[i] = stock[i]

        # Choose an owned staging point for each unresolved objective.
        fronts = {}
        for j in targets:
            if j in incoming_friendly:
                continue
            candidates = []
            for i, eta in self.adj[j].items():
                if i not in usable or i in frozen:
                    continue
                force = add(add(stock[i], training[i]), incoming_friendly.get(i, EMPTY))
                candidates.append((min(value(force), 135) * 0.15 - 6 * eta
                                   - (12 if not sum(force) else 0)
                                   - (7 if i == self.home and t > 5 else 0),
                                   -self.rank[i], i))
            if candidates:
                i = max(candidates)[2]
                objective = (target_value(j), j)
                if i not in fronts or objective[0] > fronts[i][0]:
                    fronts[i] = objective

        # Reinforce the relic before activation, and defend any exposed home core.
        hubs = dict(fronts)
        for i in (relic, self.home):
            if i not in usable or i in frozen:
                continue
            if i == relic:
                relic_force = add(stock[i], incoming_friendly.get(i, EMPTY))
                exposed = any(j in hostile for j in self.adj[i])
                if (t >= 58 or exposed) and (fighters(relic_force) < 5 or not relic_force[3]):
                    hubs[i] = (43, None)
            if any(j in hostile for j in self.adj[i]):
                hubs.setdefault(i, (23 if i == relic else 18, None))
        if not hubs:
            return orders

        # Route idle troops over owned territory, where they can be moved on again.
        distance = {i: 0.0 for i in hubs}
        next_hop = {}
        pending = set(usable)
        while pending:
            i = min(pending, key=lambda n: (distance.get(n, 1000), self.rank[n]))
            pending.remove(i)
            if i not in distance:
                break
            for j, eta in self.adj[i].items():
                if j in pending and j not in frozen and distance[i] + eta < distance.get(j, 1000):
                    distance[j] = distance[i] + eta
                    next_hop[j] = i
        for i in ordered:
            if i in next_hop and i not in frozen:
                j = next_hop[i]
                force = launchable(i, j)
                if i == self.home and t <= 10:
                    continue
                if fighters(force) and sum(incoming_friendly.get(j, EMPTY)) < 18:
                    send(i, j, force)

        # Spend only what the economy can sustain. Gold at the front can react
        # to an attack on the same turn, so keep a reserve once contact is made.
        cap = free + (max(0, income // 3) if t < 25 else max(0, income // 2))
        reserve_gold = min(60, 6 * enemy_node_count) if t >= 15 else 0
        for i, (_, j) in sorted(hubs.items(), key=lambda item: (-item[1][0], self.rank[item[0]])):
            if perf_counter() >= deadline or (population >= cap and budget < 180):
                break
            nearby = add(add(stock[i], training[i]), incoming_friendly.get(i, EMPTY))
            defense_hub = j is None
            if defense_hub:
                goal = 7 if i == relic else 4
                threat = incoming_hostile.get(i, EMPTY)
                for k in self.adj[i]:
                    if k in hostile:
                        threat = add(threat, defender(k))
                if not fighters(threat):
                    threat = with_units(EMPTY, majority, 2)
            else:
                threat = add(defender(j), incoming_hostile.get(j, EMPTY))
                eta = self.adj[i][j]
                if j in hostile:
                    goal = min(55, max(6, 3 + int(fighters(threat) * 1.9)))
                elif j == relic:
                    goal = max(5, 4 + fighters(incoming_hostile.get(j, EMPTY)))
                else:
                    goal = max(1, fighters(threat)) if j in incoming_hostile else 1
                if j in hostile and eta == 3:
                    goal += 1
            for _ in range(8):
                # The Relic garrison takes priority during the activation window.
                cash = budget if defense_hub and i == relic else budget - reserve_gold
                if cash < 10 or (population >= cap and budget < 180):
                    break
                count = fighters(nearby)
                if not nearby[3] and (count >= 3 or (goal >= 5 and count >= 2)):
                    train(i, 3, 1)
                    nearby = with_units(nearby, 3, 1)
                    continue
                if count >= goal:
                    break
                if not fighters(threat):
                    kind = (1, 2, 0)[count % 3]
                else:
                    kind = self._counter(threat)
                    if goal >= 6 and nearby[kind] >= 2 and nearby[kind] * 5 > count * 3:
                        kind = (kind + 1) % 3
                if COST[kind] > cash:
                    if kind == 2 and fighters(threat) and threat[1] == fighters(threat):
                        break
                    kind = 1 if cash >= 12 else 0
                train(i, kind, 1)
                nearby = with_units(nearby, kind, 1)
        return orders

    @staticmethod
    def _counter(force):
        r, g, s, _ = force
        scores = (r + 0.5 * g + 2 * s, 2 * r + g + 0.5 * s, 0.5 * r + 2 * g + s)
        return max(range(3), key=scores.__getitem__)
